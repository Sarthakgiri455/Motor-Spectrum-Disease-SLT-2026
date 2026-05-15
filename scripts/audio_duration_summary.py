from pathlib import Path
from collections import defaultdict

import soundfile as sf


DATA_ROOT = Path("data")
LANGUAGES = ("Spanish", "German", "Czech")
LABELS = ("HC", "PD")


def seconds_to_hms(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def wav_duration_seconds(path):
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def collect_durations():
    summary = defaultdict(lambda: {"files": 0, "seconds": 0.0})

    for language in LANGUAGES:
        language_dir = DATA_ROOT / language
        if not language_dir.exists():
            print(f"Missing language folder: {language_dir}")
            continue

        for wav_path in language_dir.rglob("*.wav"):
            parts = wav_path.parts
            label = next((part for part in parts if part in LABELS), None)
            if label is None:
                continue

            key = (language, label)
            summary[key]["files"] += 1
            summary[key]["seconds"] += wav_duration_seconds(wav_path)

    return summary


def print_summary(summary):
    print(f"{'Language':<10} {'Group':<5} {'Files':>7} {'Duration':>12} {'Hours':>10}")
    print("-" * 50)

    grand_total = {"files": 0, "seconds": 0.0}
    for language in LANGUAGES:
        for label in LABELS:
            values = summary[(language, label)]
            seconds = values["seconds"]
            grand_total["files"] += values["files"]
            grand_total["seconds"] += seconds
            print(
                f"{language:<10} {label:<5} {values['files']:>7} "
                f"{seconds_to_hms(seconds):>12} {seconds / 3600:>10.2f}"
            )

    print("-" * 50)
    print(
        f"{'TOTAL':<10} {'ALL':<5} {grand_total['files']:>7} "
        f"{seconds_to_hms(grand_total['seconds']):>12} "
        f"{grand_total['seconds'] / 3600:>10.2f}"
    )


if __name__ == "__main__":
    print_summary(collect_durations())
