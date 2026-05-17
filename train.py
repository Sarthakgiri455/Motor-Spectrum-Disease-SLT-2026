import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.msd_dataset import (
    MSDDataset,
    WhisperCollator,
    make_train_val_test_records,
)
from models.multitask_whisper import SharedWhisperMultiTaskModel


def parse_languages(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested device='cuda', but CUDA is not available.")
    return torch.device(device_name)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def compute_metrics(labels: Sequence[float], logits: Sequence[float]) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    logits_np = np.asarray(logits)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(labels_np, preds),
        "balanced_accuracy": balanced_accuracy_score(labels_np, preds),
        "f1": f1_score(labels_np, preds, zero_division=0),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "sensitivity": recall_score(labels_np, preds, zero_division=0),
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(labels_np.size),
        "num_hc": int((labels_np == 0).sum()),
        "num_pd": int((labels_np == 1).sum()),
    }
    try:
        metrics["auc"] = roc_auc_score(labels_np, probs)
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_labels: List[float] = []
    all_logits: List[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model(batch["input_features"], batch["task_id"])
            loss = criterion(logits, batch["label"])

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        total_loss += loss.item() * batch["label"].size(0)
        all_labels.extend(batch["label"].detach().cpu().tolist())
        all_logits.extend(logits.detach().cpu().tolist())

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / max(1, len(all_labels))
    return metrics


def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    all_labels: List[float] = []
    all_logits: List[float] = []
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch["input_features"], batch["task_id"])
            loss = criterion(logits, batch["label"])
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * batch["label"].size(0)
            labels = batch["label"].detach().cpu()
            logits_cpu = logits.detach().cpu()
            probs_cpu = probs.detach().cpu()
            preds_cpu = preds.detach().cpu()

            all_labels.extend(labels.tolist())
            all_logits.extend(logits_cpu.tolist())

            for index in range(labels.numel()):
                rows.append(
                    {
                        "audio_path": batch["audio_path"][index],
                        "speaker_id": batch["speaker_id"][index],
                        "language": batch["language"][index],
                        "task_type": batch["task_type"][index],
                        "label": int(labels[index].item()),
                        "logit": float(logits_cpu[index].item()),
                        "probability_pd": float(probs_cpu[index].item()),
                        "prediction": int(preds_cpu[index].item()),
                    }
                )

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / max(1, len(all_labels))
    return metrics, rows


def metrics_by_group(rows: Sequence[Dict[str, object]], group_key: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        group_value = str(row[group_key])
        grouped.setdefault(group_value, {"labels": [], "logits": []})
        grouped[group_value]["labels"].append(float(row["label"]))
        grouped[group_value]["logits"].append(float(row["logit"]))

    output = {}
    for group_value, values in grouped.items():
        output[group_value] = compute_metrics(values["labels"], values["logits"])
    return output


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_prediction_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_epoch_history(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(records, config: Dict, split: str, shuffle: bool) -> DataLoader:
    training_cfg = config["training"]
    model_cfg = config["model"]
    dataset = MSDDataset(
        records,
        sample_rate=16000,
        max_audio_seconds=training_cfg["max_audio_seconds"],
    )
    collator = WhisperCollator(
        model_name=model_cfg["name"],
        sample_rate=16000,
        max_audio_seconds=training_cfg["max_audio_seconds"],
    )
    batch_size = (
        training_cfg["batch_size"] if split == "train" else training_cfg["eval_batch_size"]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=training_cfg["num_workers"],
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def save_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([record.__dict__ for record in records], handle, indent=2)


def summarize_records(records) -> Dict[str, object]:
    summary = {
        "num_files": len(records),
        "num_speakers": len({record.speaker_id for record in records}),
        "languages": {},
        "tasks": {},
        "labels": {"HC": 0, "PD": 0},
    }
    for record in records:
        summary["languages"][record.language] = summary["languages"].get(record.language, 0) + 1
        summary["tasks"][record.task_type] = summary["tasks"].get(record.task_type, 0) + 1
        label_name = "PD" if record.label == 1 else "HC"
        summary["labels"][label_name] += 1
    return summary


def summarize_model_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    encoder_total = sum(parameter.numel() for parameter in model.encoder.parameters())
    encoder_trainable = sum(
        parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "encoder_total_parameters": encoder_total,
        "encoder_trainable_parameters": encoder_trainable,
        "head_trainable_parameters": trainable - encoder_trainable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train_languages", required=True)
    parser.add_argument("--test_languages", required=True)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    train_languages = parse_languages(args.train_languages)
    test_languages = parse_languages(args.test_languages)
    run_name = args.run_name or (
        f"train_{'-'.join(train_languages)}__test_{'-'.join(test_languages)}"
    )
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, val_records, test_records = make_train_val_test_records(
        data_root=config["data_root"],
        train_languages=train_languages,
        test_languages=test_languages,
        validation_size=config["training"]["validation_size"],
        seed=config["seed"],
    )
    save_records(output_dir / "train_records.json", train_records)
    save_records(output_dir / "val_records.json", val_records)
    save_records(output_dir / "test_records.json", test_records)
    run_metadata = {
        "run_name": run_name,
        "config_path": args.config,
        "train_languages": train_languages,
        "test_languages": test_languages,
        "seed": config["seed"],
        "config": config,
        "splits": {
            "train": summarize_records(train_records),
            "validation": summarize_records(val_records),
            "test": summarize_records(test_records),
        },
    }
    write_json(output_dir / "run_config.json", run_metadata)
    write_yaml(output_dir / "run_config.yaml", run_metadata)

    train_loader = make_loader(train_records, config, split="train", shuffle=True)
    val_loader = make_loader(val_records, config, split="val", shuffle=False)
    test_loader = make_loader(test_records, config, split="test", shuffle=False)

    device = resolve_device(config.get("device", "auto"))
    print(f"device={device}")
    model = SharedWhisperMultiTaskModel(
        model_name=config["model"]["name"],
        freeze_encoder=config["model"]["freeze_encoder"],
        unfreeze_last_n_layers=config["model"]["unfreeze_last_n_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    parameter_summary = summarize_model_parameters(model)
    write_json(output_dir / "model_summary.json", parameter_summary)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir=str(Path(config["log_dir"]) / run_name))

    best_f1 = -1.0
    best_path = output_dir / "best_model.pt"
    history: List[Dict[str, object]] = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=config["training"]["gradient_clip_norm"],
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"validation/{key}", value, epoch)

        epoch_row = {"epoch": epoch}
        epoch_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        epoch_row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(epoch_row)
        write_epoch_history(output_dir / "epoch_metrics.csv", history)
        write_json(output_dir / "epoch_metrics.json", history)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} train_auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "train_languages": train_languages,
                    "test_languages": test_languages,
                    "best_validation_f1": best_f1,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics, train_predictions = predict(model, train_loader, criterion, device)
    val_metrics, val_predictions = predict(model, val_loader, criterion, device)
    test_metrics, test_predictions = predict(model, test_loader, criterion, device)
    writer.close()

    final_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "test_by_task": metrics_by_group(test_predictions, "task_type"),
        "test_by_language": metrics_by_group(test_predictions, "language"),
    }
    write_json(output_dir / "final_metrics.json", final_metrics)
    write_json(output_dir / "train_metrics.json", train_metrics)
    write_json(output_dir / "val_metrics.json", val_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_json(output_dir / "test_metrics_by_task.json", final_metrics["test_by_task"])
    write_json(output_dir / "test_metrics_by_language.json", final_metrics["test_by_language"])
    write_prediction_csv(output_dir / "train_predictions.csv", train_predictions)
    write_prediction_csv(output_dir / "val_predictions.csv", val_predictions)
    write_prediction_csv(output_dir / "test_predictions.csv", test_predictions)
    print(f"best_checkpoint={best_path}")
    print(f"test_metrics={json.dumps(test_metrics, indent=2)}")


if __name__ == "__main__":
    main()
