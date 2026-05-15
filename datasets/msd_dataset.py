import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor


TASK_TO_ID = {"ddk": 0, "vowel": 1, "read": 2}
ID_TO_TASK = {v: k for k, v in TASK_TO_ID.items()}
LABEL_TO_ID = {"HC": 0, "PD": 1, "MSD": 1}
LANGUAGE_TO_ID = {"Spanish": 0, "German": 1, "Czech": 2}
SPEAKER_RE = re.compile(r"^(?P<speaker>[a-z]{3}_(?:HC|PD)_\d+)")


@dataclass(frozen=True)
class MSDRecord:
    audio_path: str
    language: str
    task_type: str
    label: int
    speaker_id: str

    @property
    def language_id(self) -> int:
        return LANGUAGE_TO_ID[self.language]

    @property
    def task_id(self) -> int:
        return TASK_TO_ID[self.task_type]


def parse_speaker_id(path: Path) -> str:
    match = SPEAKER_RE.match(path.stem)
    if not match:
        raise ValueError(f"Could not parse speaker id from filename: {path.name}")
    return match.group("speaker")


def infer_task_type(parts: Sequence[str]) -> Optional[str]:
    for task in TASK_TO_ID:
        if task in parts:
            return task
    return None


def build_records(
    data_root: str = "data",
    languages: Optional[Iterable[str]] = None,
) -> List[MSDRecord]:
    root = Path(data_root)
    selected_languages = set(languages) if languages else set(LANGUAGE_TO_ID)
    records: List[MSDRecord] = []

    for language in sorted(selected_languages):
        language_dir = root / language
        if not language_dir.exists():
            raise FileNotFoundError(f"Missing language directory: {language_dir}")

        for wav_path in sorted(language_dir.rglob("*.wav")):
            rel_parts = wav_path.relative_to(language_dir).parts
            label = next((part for part in rel_parts if part in LABEL_TO_ID), None)
            task_type = infer_task_type(rel_parts)
            if label is None or task_type is None:
                continue

            records.append(
                MSDRecord(
                    audio_path=str(wav_path),
                    language=language,
                    task_type=task_type,
                    label=LABEL_TO_ID[label],
                    speaker_id=parse_speaker_id(wav_path),
                )
            )

    return records


def speaker_split(
    records: Sequence[MSDRecord],
    validation_size: float,
    seed: int,
) -> Tuple[List[MSDRecord], List[MSDRecord]]:
    speaker_to_records: Dict[str, List[MSDRecord]] = {}
    for record in records:
        speaker_to_records.setdefault(record.speaker_id, []).append(record)

    speakers = sorted(speaker_to_records)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_val = max(1, round(len(speakers) * validation_size)) if speakers else 0
    val_speakers = set(speakers[:n_val])

    train_records = []
    val_records = []
    for speaker_id, speaker_records in speaker_to_records.items():
        if speaker_id in val_speakers:
            val_records.extend(speaker_records)
        else:
            train_records.extend(speaker_records)

    return train_records, val_records


def make_train_val_test_records(
    data_root: str,
    train_languages: Sequence[str],
    test_languages: Sequence[str],
    validation_size: float,
    seed: int,
) -> Tuple[List[MSDRecord], List[MSDRecord], List[MSDRecord]]:
    all_languages = sorted(set(train_languages) | set(test_languages))
    all_records = build_records(data_root=data_root, languages=all_languages)

    train_pool = [r for r in all_records if r.language in train_languages]
    test_records = [r for r in all_records if r.language in test_languages]

    if set(train_languages) & set(test_languages):
        test_speakers = {r.speaker_id for r in test_records}
        train_pool = [r for r in train_pool if r.speaker_id not in test_speakers]

    train_records, val_records = speaker_split(
        train_pool, validation_size=validation_size, seed=seed
    )
    assert_speaker_disjoint(train_records, val_records, "train", "validation")
    assert_speaker_disjoint(train_records, test_records, "train", "test")
    return train_records, val_records, test_records


def assert_speaker_disjoint(
    left: Sequence[MSDRecord],
    right: Sequence[MSDRecord],
    left_name: str,
    right_name: str,
) -> None:
    overlap = {r.speaker_id for r in left} & {r.speaker_id for r in right}
    if overlap:
        examples = ", ".join(sorted(overlap)[:10])
        raise ValueError(
            f"{left_name}/{right_name} speaker overlap detected: {examples}"
        )


class MSDDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MSDRecord],
        sample_rate: int = 16000,
        max_audio_seconds: Optional[float] = 30.0,
    ) -> None:
        self.records = list(records)
        self.sample_rate = sample_rate
        self.max_audio_samples = (
            int(sample_rate * max_audio_seconds) if max_audio_seconds else None
        )
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == self.sample_rate:
            return waveform
        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                sample_rate, self.sample_rate
            )
        return self._resamplers[sample_rate](waveform)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        waveform, sample_rate = torchaudio.load(record.audio_path)
        waveform = waveform.mean(dim=0)
        waveform = self._resample(waveform, sample_rate)

        if self.max_audio_samples and waveform.numel() > self.max_audio_samples:
            waveform = waveform[: self.max_audio_samples]

        waveform = waveform.float()
        waveform = waveform - waveform.mean()
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return {
            "waveform": waveform,
            "task_id": torch.tensor(record.task_id, dtype=torch.long),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "language_id": torch.tensor(record.language_id, dtype=torch.long),
            "speaker_id": record.speaker_id,
            "audio_path": record.audio_path,
            "language": record.language,
            "task_type": record.task_type,
        }


class WhisperCollator:
    def __init__(
        self,
        model_name: str,
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
    ) -> None:
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.max_audio_samples = int(sample_rate * max_audio_seconds)

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        waveforms = []
        for item in batch:
            waveform = item["waveform"]
            if waveform.numel() > self.max_audio_samples:
                waveform = waveform[: self.max_audio_samples]
            waveforms.append(waveform.numpy())

        features = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_audio_samples,
        )

        return {
            "input_features": features.input_features,
            "task_id": torch.stack([item["task_id"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "language_id": torch.stack([item["language_id"] for item in batch]),
            "speaker_id": [item["speaker_id"] for item in batch],
            "audio_path": [item["audio_path"] for item in batch],
            "language": [item["language"] for item in batch],
            "task_type": [item["task_type"] for item in batch],
        }

