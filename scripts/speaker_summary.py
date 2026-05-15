import re
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path("data")
LANGUAGES = ("Spanish", "German", "Czech")
LABELS = ("HC", "PD")
SPEAKER_RE = re.compile(r"^(?P<prefix>[a-z]{3}_(?:HC|PD)_\d+)")


def parse_speaker_id(wav_path):
    match = SPEAKER_RE.match(wav_path.stem)
    if not match:
        return None
    return match.group("prefix")


def collect_speakers():
    speakers_by_group = defaultdict(set)
    speakers_by_task = defaultdict(set)
    files_by_task = defaultdict(int)
    unparsed = []

    for language in LANGUAGES:
        language_dir = DATA_ROOT / language
        if not language_dir.exists():
            print(f"Missing language folder: {language_dir}")
            continue

        for wav_path in language_dir.rglob("*.wav"):
            relative_parts = wav_path.relative_to(language_dir).parts
            label = next((part for part in relative_parts if part in LABELS), None)
            if label is None:
                continue

            speaker_id = parse_speaker_id(wav_path)
            if speaker_id is None:
                unparsed.append(wav_path)
                continue

            label_index = relative_parts.index(label)
            task = "/".join(relative_parts[:label_index])

            speakers_by_group[(language, label)].add(speaker_id)
            speakers_by_task[(language, task, label)].add(speaker_id)
            files_by_task[(language, task, label)] += 1

    return speakers_by_group, speakers_by_task, files_by_task, unparsed


def print_group_summary(speakers_by_group):
    print("Unique Speakers By Language And Group")
    print(f"{'Language':<10} {'Group':<5} {'Speakers':>9}")
    print("-" * 28)
    for language in LANGUAGES:
        for label in LABELS:
            print(f"{language:<10} {label:<5} {len(speakers_by_group[(language, label)]):>9}")


def print_task_summary(speakers_by_task, files_by_task):
    print("\nUnique Speakers By Task")
    print(f"{'Language':<10} {'Task':<14} {'Group':<5} {'Speakers':>9} {'Files':>7}")
    print("-" * 52)
    for language in LANGUAGES:
        keys = sorted(k for k in speakers_by_task if k[0] == language)
        for _, task, label in keys:
            print(
                f"{language:<10} {task:<14} {label:<5} "
                f"{len(speakers_by_task[(language, task, label)]):>9} "
                f"{files_by_task[(language, task, label)]:>7}"
            )


if __name__ == "__main__":
    speakers_by_group, speakers_by_task, files_by_task, unparsed = collect_speakers()
    print_group_summary(speakers_by_group)
    print_task_summary(speakers_by_task, files_by_task)

    if unparsed:
        print("\nUnparsed files:")
        for path in unparsed:
            print(path)
