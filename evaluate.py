import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.msd_dataset import MSDDataset, WhisperCollator, build_records
from models.multitask_whisper import SharedWhisperMultiTaskModel
from train import (
    metrics_by_group,
    parse_languages,
    predict,
    resolve_device,
    write_json,
    write_prediction_csv,
)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--test_languages", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_predictions_csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    test_languages = parse_languages(args.test_languages)
    records = build_records(config["data_root"], languages=test_languages)

    dataset = MSDDataset(
        records,
        sample_rate=16000,
        max_audio_seconds=config["training"]["max_audio_seconds"],
    )
    collator = WhisperCollator(
        model_name=config["model"]["name"],
        sample_rate=16000,
        max_audio_seconds=config["training"]["max_audio_seconds"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(config.get("device", "auto"))
    print(f"device={device}")
    model = SharedWhisperMultiTaskModel(
        model_name=config["model"]["name"],
        freeze_encoder=config["model"]["freeze_encoder"],
        unfreeze_last_n_layers=config["model"]["unfreeze_last_n_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.BCEWithLogitsLoss()
    metrics, predictions = predict(model, loader, criterion, device)
    metrics_payload = {
        "overall": metrics,
        "by_task": metrics_by_group(predictions, "task_type"),
        "by_language": metrics_by_group(predictions, "language"),
    }
    print(json.dumps(metrics_payload, indent=2))

    if args.output_json:
        write_json(Path(args.output_json), metrics_payload)

    if args.output_predictions_csv:
        write_prediction_csv(Path(args.output_predictions_csv), predictions)


if __name__ == "__main__":
    main()
