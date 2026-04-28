"""Inspect DEAP original BDF channel/event structure for selected subjects.

This script is evidence-only: it does not modify preprocessing logic. It first
tries to use MNE when available. If MNE is not installed, it falls back to a
small BDF header parser and scans the last channel as a possible event channel.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "data/DEAP/original"
RESULT_PATH = (
    PROJECT_ROOT
    / "results/preprocessing_all_subjects/channel_inspection_report.txt"
)
FILES_TO_CHECK = ("s01.bdf", "s23.bdf", "s24.bdf", "s25.bdf", "s32.bdf")
STATUS_KEYWORDS = ("status",)
EVENT_KEYWORDS = ("trigger", "stim", "sti", "annotation", "event", "status")


def decode_ascii(raw_bytes: bytes) -> str:
    return raw_bytes.decode("ascii", errors="ignore").strip()


def read_bdf_header(path: Path) -> dict:
    """Read BDF header fields needed for channel inspection."""
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


def infer_channel_type(index: int, name: str, num_signals: int) -> str:
    """Fallback channel type guess when MNE is not available."""
    normalized = name.strip().lower()
    if any(keyword in normalized for keyword in EVENT_KEYWORDS):
        return "stim/event-like"
    if index < 32:
        return "eeg"
    if index == num_signals - 1:
        return "last-channel-unknown"
    return "misc/physio"


def find_status_like_channels(channel_names: list[str]) -> dict:
    normalized = [name.strip().lower() for name in channel_names]
    return {
        "exact_status": [name for name in channel_names if name == "Status"],
        "case_or_space_status": [
            channel_names[index]
            for index, name in enumerate(normalized)
            if any(keyword in name for keyword in STATUS_KEYWORDS)
        ],
        "event_like": [
            channel_names[index]
            for index, name in enumerate(normalized)
            if any(keyword in name for keyword in EVENT_KEYWORDS)
        ],
    }


def scan_last_channel(path: Path, header: dict, max_changes: int = 30) -> dict:
    """Scan the final BDF channel as a possible event/status channel."""
    samples_per_record = header["samples_per_record"]
    last_index = header["num_signals"] - 1
    record_bytes = sum(samples_per_record) * 3
    channel_offset = sum(samples_per_record[:last_index]) * 3
    channel_bytes = samples_per_record[last_index] * 3
    sampling_rate = header["sampling_rates"][last_index]

    raw_counter = Counter()
    low8_counter = Counter()
    min_value = None
    max_value = None
    changes = []
    previous = None
    sample_index = 0

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
                min_value = value if min_value is None else min(min_value, value)
                max_value = value if max_value is None else max(max_value, value)

                if previous is not None and value != previous and len(changes) < max_changes:
                    changes.append(
                        {
                            "sample": sample_index,
                            "time_seconds": sample_index / sampling_rate,
                            "from": previous,
                            "to": value,
                            "from_low8": previous & 0xFF,
                            "to_low8": low8,
                        }
                    )
                previous = value
                sample_index += 1

    possible_event = (
        len(raw_counter) > 1
        and len(raw_counter) < max(200, sample_index * 0.01)
    ) or len(low8_counter) > 1

    return {
        "name": header["labels"][last_index],
        "sampling_rate": sampling_rate,
        "unique_raw_count": len(raw_counter),
        "unique_low8_count": len(low8_counter),
        "min": min_value,
        "max": max_value,
        "most_common_raw": raw_counter.most_common(12),
        "most_common_low8": low8_counter.most_common(12),
        "changes": changes,
        "possible_event_channel": possible_event,
    }


def try_mne_inspection(path: Path) -> dict | None:
    """Return MNE inspection data if MNE is installed, otherwise None."""
    try:
        import mne
    except ImportError:
        return None

    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    return {
        "sfreq": raw.info["sfreq"],
        "ch_names": list(raw.ch_names),
        "channel_types": raw.get_channel_types(),
        "annotations": list(raw.annotations),
    }


def format_channel_table(channel_names: list[str], channel_types: list[str]) -> list[str]:
    return [
        f"  {index:02d}: {name} | type={channel_types[index]}"
        for index, name in enumerate(channel_names)
    ]


def inspect_one_file(file_name: str) -> str:
    path = ORIGINAL_DIR / file_name
    lines = [
        "=" * 80,
        f"File: {file_name}",
        f"Path: {path}",
        f"Exists: {path.exists()}",
    ]

    if not path.exists():
        return "\n".join(lines)

    lines.append(f"File size bytes: {path.stat().st_size}")
    mne_info = try_mne_inspection(path)
    header = read_bdf_header(path)

    if mne_info is not None:
        channel_names = mne_info["ch_names"]
        channel_types = mne_info["channel_types"]
        lines.append("Reader: MNE")
        lines.append(f'raw.info["sfreq"]: {mne_info["sfreq"]}')
        lines.append(f"len(raw.ch_names): {len(channel_names)}")
        annotations = mne_info["annotations"]
        lines.append(f"raw.annotations count: {len(annotations)}")
        lines.append("first annotations:")
        for annotation in annotations[:5]:
            lines.append(f"  {annotation}")
    else:
        channel_names = header["labels"]
        channel_types = [
            infer_channel_type(index, name, header["num_signals"])
            for index, name in enumerate(channel_names)
        ]
        lines.append("Reader: fallback BDF header parser (MNE not installed)")
        lines.append(f'raw.info["sfreq"] equivalent: {header["sampling_rates"][0]}')
        lines.append(f"len(raw.ch_names) equivalent: {len(channel_names)}")
        lines.append("raw.annotations count: unavailable without MNE")
        lines.append("first annotations: unavailable without MNE")

    matches = find_status_like_channels(channel_names)
    lines.append("Complete channel names and inferred/types:")
    lines.extend(format_channel_table(channel_names, channel_types))
    lines.append(f"Last 5 channel names: {channel_names[-5:]}")
    lines.append(f"Exact Status channel exists: {bool(matches['exact_status'])}")
    lines.append(f"Status-like channels: {matches['case_or_space_status']}")
    lines.append(f"Trigger/Stim/STI/Annotation/Event-like channels: {matches['event_like']}")

    if not matches["exact_status"]:
        last_scan = scan_last_channel(path, header)
        lines.append("Last-channel event-candidate scan:")
        lines.append(f"  last channel name: {last_scan['name']}")
        lines.append(f"  sampling rate: {last_scan['sampling_rate']}")
        lines.append(f"  unique raw values: {last_scan['unique_raw_count']}")
        lines.append(f"  unique low8 values: {last_scan['unique_low8_count']}")
        lines.append(f"  min/max raw values: {last_scan['min']} / {last_scan['max']}")
        lines.append(f"  most common raw values: {last_scan['most_common_raw']}")
        lines.append(f"  most common low8 values: {last_scan['most_common_low8']}")
        lines.append("  first value changes:")
        for change in last_scan["changes"]:
            lines.append(
                "    sample={sample}, time={time_seconds:.3f}s, "
                "raw {from}->{to}, low8 {from_low8}->{to_low8}".format(**change)
            )
        lines.append(
            "  possible event channel by simple evidence: "
            f"{last_scan['possible_event_channel']}"
        )

    return "\n".join(lines)


def main() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [inspect_one_file(file_name) for file_name in FILES_TO_CHECK]
    RESULT_PATH.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"saved channel inspection report to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
