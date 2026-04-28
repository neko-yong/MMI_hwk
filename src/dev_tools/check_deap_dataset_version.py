"""Compare current DEAP original data with a Kaggle DEAP copy.

This script is read-only. It does not modify preprocessing logic or overwrite
data. It checks representative BDF files to decide which dataset layout better
matches the current preprocess.py assumptions.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRS = {
    "current": PROJECT_ROOT / "data/DEAP/original",
    "kaggle": PROJECT_ROOT / "data/DEAP_kaggle/original",
}
FILES_TO_CHECK = ("s01.bdf", "s24.bdf", "s32.bdf")
REPORT_PATH = (
    PROJECT_ROOT
    / "results/dataset_version_check/deap_dataset_version_report.txt"
)


def decode_ascii(raw_bytes: bytes) -> str:
    return raw_bytes.decode("ascii", errors="ignore").strip()


def read_bdf_header(path: Path) -> dict:
    with path.open("rb") as file:
        fixed = file.read(256)

    header_bytes = int(decode_ascii(fixed[184:192]))
    num_records = int(decode_ascii(fixed[236:244]))
    record_duration = float(decode_ascii(fixed[244:252]))
    num_signals = int(decode_ascii(fixed[252:256]))

    with path.open("rb") as file:
        header = file.read(header_bytes)

    offset = 256
    labels = [
        decode_ascii(header[offset + index * 16 : offset + (index + 1) * 16])
        for index in range(num_signals)
    ]

    offset += num_signals * 16
    offset += num_signals * 80
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 80
    samples_per_record = [
        int(decode_ascii(header[offset + index * 8 : offset + (index + 1) * 8]))
        for index in range(num_signals)
    ]
    sampling_rates = [samples / record_duration for samples in samples_per_record]

    return {
        "header_bytes": header_bytes,
        "num_records": num_records,
        "record_duration": record_duration,
        "num_signals": num_signals,
        "labels": labels,
        "samples_per_record": samples_per_record,
        "sampling_rates": sampling_rates,
    }


def read_uint24_values(raw_bytes: bytes):
    for index in range(0, len(raw_bytes), 3):
        yield (
            raw_bytes[index]
            | (raw_bytes[index + 1] << 8)
            | (raw_bytes[index + 2] << 16)
        )


def scan_channel_events(path: Path, header: dict, channel_index: int) -> dict:
    """Scan one channel and count low-8-bit rising events."""
    samples_per_record = header["samples_per_record"]
    record_bytes = sum(samples_per_record) * 3
    channel_offset = sum(samples_per_record[:channel_index]) * 3
    channel_bytes = samples_per_record[channel_index] * 3
    sampling_rate = header["sampling_rates"][channel_index]
    raw_counter = Counter()
    low8_counter = Counter()
    rising_counts = Counter()
    previous = None
    sample_index = 0
    first_changes = []

    with path.open("rb") as file:
        for record_index in range(header["num_records"]):
            file.seek(
                header["header_bytes"]
                + record_index * record_bytes
                + channel_offset
            )
            raw_channel = file.read(channel_bytes)

            for value in read_uint24_values(raw_channel):
                low8 = value & 0xFF
                raw_counter[value] += 1
                low8_counter[low8] += 1

                if previous is not None and value != previous:
                    previous_low8 = previous & 0xFF
                    if previous_low8 == 0 and low8 != 0:
                        rising_counts[low8] += 1
                    if len(first_changes) < 20:
                        first_changes.append(
                            (
                                sample_index,
                                sample_index / sampling_rate,
                                previous,
                                value,
                                previous_low8,
                                low8,
                            )
                        )
                previous = value
                sample_index += 1

    can_detect_345 = all(rising_counts.get(code, 0) >= 40 for code in (3, 4, 5))
    can_recover_40_trials = all(rising_counts.get(code, 0) == 40 for code in (3, 4, 5))
    possible_event = (
        len(raw_counter) > 1
        and len(raw_counter) < max(200, sample_index * 0.01)
    ) or len(low8_counter) > 1

    return {
        "channel_name": header["labels"][channel_index],
        "channel_index": channel_index,
        "possible_event_channel": possible_event,
        "unique_raw_count": len(raw_counter),
        "unique_low8_count": len(low8_counter),
        "most_common_low8": low8_counter.most_common(12),
        "rising_counts": dict(sorted(rising_counts.items())),
        "can_detect_code_3_4_5": can_detect_345,
        "can_recover_40_trials": can_recover_40_trials,
        "first_changes": first_changes,
    }


def choose_event_channel_index(header: dict) -> int:
    """Prefer exact Status, otherwise inspect the final channel."""
    labels = header["labels"]
    if "Status" in labels:
        return labels.index("Status")
    return len(labels) - 1


def inspect_file(dataset_name: str, directory: Path, file_name: str) -> dict:
    path = directory / file_name
    result = {
        "dataset": dataset_name,
        "file": file_name,
        "path": path,
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    header = read_bdf_header(path)
    labels = header["labels"]
    event_index = choose_event_channel_index(header)
    event_scan = scan_channel_events(path, header, event_index)
    result.update(
        {
            "file_size": path.stat().st_size,
            "sampling_rate": header["sampling_rates"][0],
            "channel_count": len(labels),
            "channel_names": labels,
            "last_5_channels": labels[-5:],
            "has_status": "Status" in labels,
            "has_empty_channel_name": any(name == "" for name in labels),
            "event_channel_checked": event_scan["channel_name"],
            "event_channel_index": event_scan["channel_index"],
            "last_channel_event_like": event_scan["possible_event_channel"],
            "rising_counts": event_scan["rising_counts"],
            "can_detect_code_3_4_5": event_scan["can_detect_code_3_4_5"],
            "can_recover_40_trials": event_scan["can_recover_40_trials"],
            "event_scan": event_scan,
        }
    )
    return result


def format_file_report(result: dict) -> list[str]:
    lines = [
        "-" * 80,
        f"Dataset: {result['dataset']}",
        f"File: {result['file']}",
        f"Path: {result['path']}",
        f"Exists: {result['exists']}",
    ]

    if not result["exists"]:
        return lines

    lines.extend(
        [
            f"File size: {result['file_size']}",
            f"Sampling rate: {result['sampling_rate']}",
            f"Channel count: {result['channel_count']}",
            f"Complete channel names: {result['channel_names']}",
            f"Last 5 channel names: {result['last_5_channels']}",
            f"Has Status channel: {result['has_status']}",
            f"Has empty channel name: {result['has_empty_channel_name']}",
            f"Event channel checked: index={result['event_channel_index']}, "
            f"name={result['event_channel_checked']!r}",
            f"Last/event channel looks event-like: {result['last_channel_event_like']}",
            f"Rising low8 event counts: {result['rising_counts']}",
            f"Can detect code 3/4/5: {result['can_detect_code_3_4_5']}",
            f"Can recover 40 trials: {result['can_recover_40_trials']}",
            f"Most common low8 values: {result['event_scan']['most_common_low8']}",
            "First channel value changes:",
        ]
    )
    for change in result["event_scan"]["first_changes"]:
        lines.append(
            "  sample={}, time={:.3f}s, raw {}->{}, low8 {}->{}".format(*change)
        )

    return lines


def summarize_dataset(results: list[dict]) -> list[str]:
    lines = [
        "=" * 80,
        "Dataset-level summary",
    ]

    for dataset_name in DATASET_DIRS:
        dataset_results = [item for item in results if item["dataset"] == dataset_name]
        existing = [item for item in dataset_results if item["exists"]]
        status_count = sum(1 for item in existing if item.get("has_status"))
        recover_count = sum(1 for item in existing if item.get("can_recover_40_trials"))
        empty_name_count = sum(1 for item in existing if item.get("has_empty_channel_name"))
        lines.extend(
            [
                "",
                f"{dataset_name}:",
                f"  checked files: {len(dataset_results)}",
                f"  existing files: {len(existing)}",
                f"  files with Status channel: {status_count}",
                f"  files with empty channel names: {empty_name_count}",
                f"  files where checked event channel recovers 40 trials: {recover_count}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- Current preprocess.py expects an exact 'Status' channel.",
            "- A dataset version is better aligned with the current logic if all "
            "representative files have Status and recover 40 trials via codes 3/4/5.",
            "- Empty channel names plus event-like final-channel values suggest a "
            "structural naming difference rather than necessarily missing events.",
        ]
    )

    return lines


def main() -> None:
    results = []
    for dataset_name, directory in DATASET_DIRS.items():
        for file_name in FILES_TO_CHECK:
            print(f"checking {dataset_name}: {file_name}")
            results.append(inspect_file(dataset_name, directory, file_name))

    lines = summarize_dataset(results)
    for result in results:
        lines.extend(format_file_report(result))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved dataset version report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
