# Multilingual Multi-Task MSD Classification

This repository contains a PyTorch pipeline for binary motor speech disorder
(MSD) classification from multilingual speech. Each audio file is treated as an
independent sample. Recordings are not concatenated.

The dataset is inferred from the directory hierarchy:

```text
data/<Language>/<task>/<HC|PD>/*.wav
data/Spanish/extended/<task>/<HC|PD>/*.wav
```

Supported languages are `Spanish`, `German`, and `Czech`. Supported task types
are `ddk`, `vowel`, and `read`. All sustained vowel recordings are mapped to
`task_type="vowel"` regardless of the vowel token in the filename.

## Model

The deep learning baseline uses a shared Hugging Face Whisper encoder with
three task-specific binary classification heads:

- `ddk_head`
- `vowel_head`
- `read_head`

The decoder is not used. Whisper encoder frame representations are mean-pooled
over time, then routed to the correct task head inside mixed-task batches.

## Files

```text
configs/base.yaml                 Training configuration
datasets/msd_dataset.py           Dataset scanning, audio loading, splits, collator
models/multitask_whisper.py       Shared Whisper encoder + task heads
train.py                          Training CLI
evaluate.py                       Evaluation CLI
```

## Install

```bash
pip install -r requirements.txt
```

If you already use the local virtual environment:

```bash
./venv/bin/pip install -r requirements.txt
```

## Train

Experiment A, train on Spanish and test on German:

```bash
python train.py \
  --config configs/base.yaml \
  --train_languages Spanish \
  --test_languages German \
  --run_name spanish_to_german
```

Experiment A, train on Spanish and test on Czech:

```bash
python train.py \
  --config configs/base.yaml \
  --train_languages Spanish \
  --test_languages Czech \
  --run_name spanish_to_czech
```

Experiment B, train on Spanish + Czech and test on German:

```bash
python train.py \
  --config configs/base.yaml \
  --train_languages Spanish,Czech \
  --test_languages German \
  --run_name spanish_czech_to_german
```

The best checkpoint is saved by validation F1 under:

```text
checkpoints/<run_name>/best_model.pt
```

TensorBoard logs are written under:

```text
runs/<run_name>/
```

Launch TensorBoard with:

```bash
tensorboard --logdir runs
```

## Evaluate

Evaluate a trained checkpoint on one or more languages:

```bash
python evaluate.py \
  --config configs/base.yaml \
  --checkpoint checkpoints/spanish_to_german/best_model.pt \
  --test_languages German
```

For leave-one-language-out transfer, set `--train_languages` to two languages
and `--test_languages` to the held-out language.

## Speaker Independence

Speaker IDs are parsed from filenames such as `spa_PD_063_read.wav` as
`spa_PD_063`. The training script creates a speaker-disjoint validation split
from the training languages and checks that train, validation, and test speakers
do not overlap.
