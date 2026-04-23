"""Minimal DEAP raw data inspection utilities.

This module intentionally does not perform filtering, artifact removal,
segmentation, feature extraction, or modeling. It only checks whether the raw
DEAP BDF files and metadata CSV files can be read and understood.
"""

from __future__ import annotations

import csv
from array import array
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_DIR = Path("data/DEAP/original")
DEFAULT_METADATA_DIR = Path("data/DEAP/metadata")

EEG_CHANNEL_COUNT = 32
DEAP_TRIAL_DURATION_SECONDS = 60
DEAP_BASELINE_DURATION_SECONDS = 5
EXPECTED_DEAP_TRIAL_COUNT = 40
DEAP_SAMPLING_RATE = 512
TARGET_BASELINE_SAMPLES = DEAP_BASELINE_DURATION_SECONDS * DEAP_SAMPLING_RATE
TARGET_STIMULUS_SAMPLES = DEAP_TRIAL_DURATION_SECONDS * DEAP_SAMPLING_RATE
STATUS_CHANNEL_NAME = "Status"
STATUS_EVENT_MASK = 0xFF
EVENT_BASELINE_START = 3
EVENT_STIMULUS_START = 4
EVENT_STIMULUS_END = 5

# Conservative defaults for course demonstration:
# - 4-45 Hz keeps common EEG emotion-related bands while reducing drift/noise.
# - 50 Hz notch suppresses mains interference in China.
BANDPASS_LOW_HZ = 4.0
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4
NOTCH_FREQ_HZ = 50.0
NOTCH_QUALITY_FACTOR = 30.0


def preprocess_eeg(eeg_data):
    """Return EEG data without modification as a placeholder step."""
    return eeg_data


def _decode_ascii(raw_bytes: bytes) -> str:
    return raw_bytes.decode("ascii", errors="ignore").strip()


def read_bdf_header(bdf_path: str | Path) -> dict:
    """Read only the BDF header and return basic structure information."""
    path = Path(bdf_path)
    with path.open("rb") as file:
        fixed_header = file.read(256)

    header_bytes = int(_decode_ascii(fixed_header[184:192]))
    num_records = int(_decode_ascii(fixed_header[236:244]))
    record_duration = float(_decode_ascii(fixed_header[244:252]))
    num_signals = int(_decode_ascii(fixed_header[252:256]))

    with path.open("rb") as file:
        header = file.read(header_bytes)

    offset = 256
    labels = [
        _decode_ascii(header[offset + i * 16 : offset + (i + 1) * 16])
        for i in range(num_signals)
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
        int(_decode_ascii(header[offset + i * 8 : offset + (i + 1) * 8]))
        for i in range(num_signals)
    ]

    sampling_rates = [
        samples / record_duration for samples in samples_per_record
    ]

    return {
        "file_name": path.name,
        "data_type": "BioSemi BDF raw continuous recording",
        "version": _decode_ascii(fixed_header[0:8]),
        "participant": _decode_ascii(fixed_header[8:88]),
        "header_bytes": header_bytes,
        "num_records": num_records,
        "record_duration_seconds": record_duration,
        "num_signals": num_signals,
        "channel_labels": labels,
        "samples_per_record": samples_per_record,
        "sampling_rates": sampling_rates,
        "total_samples_per_channel": num_records * samples_per_record[0],
    }


def _read_uint24_samples(raw_bytes: bytes) -> list[int]:
    """Decode little-endian 24-bit BDF samples as unsigned integers."""
    samples = []
    for index in range(0, len(raw_bytes), 3):
        samples.append(
            raw_bytes[index]
            | (raw_bytes[index + 1] << 8)
            | (raw_bytes[index + 2] << 16)
        )
    return samples


def _append_signed_int24_samples(
    target: array,
    raw_bytes: bytes,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> None:
    """Append little-endian signed 24-bit BDF samples to an integer array."""
    if end_sample is None:
        end_sample = len(raw_bytes) // 3

    for sample_index in range(start_sample, end_sample):
        byte_index = sample_index * 3
        value = (
            raw_bytes[byte_index]
            | (raw_bytes[byte_index + 1] << 8)
            | (raw_bytes[byte_index + 2] << 16)
        )
        if value & 0x800000:
            value -= 0x1000000
        target.append(value)


def extract_status_events(bdf_path: str | Path) -> dict:
    """Extract normalized events from the BDF status channel.

    Only the status channel is scanned. EEG signal channels are not loaded.
    BioSemi status values contain high-bit device state information, so this
    function keeps the low 8 bits as event codes.
    """
    path = Path(bdf_path)
    header = read_bdf_header(path)
    channel_labels = header["channel_labels"]

    if STATUS_CHANNEL_NAME not in channel_labels:
        raise ValueError(f"Cannot find status channel: {STATUS_CHANNEL_NAME}")

    status_index = channel_labels.index(STATUS_CHANNEL_NAME)
    samples_per_record = header["samples_per_record"]
    record_bytes = sum(samples_per_record) * 3
    status_offset_in_record = sum(samples_per_record[:status_index]) * 3
    status_bytes_per_record = samples_per_record[status_index] * 3
    sampling_rate = header["sampling_rates"][status_index]

    events = []
    event_value_counts = Counter()
    rising_event_counts = Counter()
    previous_value = None
    sample_index = 0

    with path.open("rb") as file:
        for record_index in range(header["num_records"]):
            file.seek(
                header["header_bytes"]
                + record_index * record_bytes
                + status_offset_in_record
            )
            raw_status = file.read(status_bytes_per_record)

            for raw_value in _read_uint24_samples(raw_status):
                event_value = raw_value & STATUS_EVENT_MASK
                event_value_counts[event_value] += 1

                if previous_value is None:
                    previous_value = event_value
                elif event_value != previous_value:
                    if previous_value == 0 and event_value != 0:
                        events.append(
                            {
                                "sample": sample_index,
                                "time_seconds": sample_index / sampling_rate,
                                "event_code": event_value,
                            }
                        )
                        rising_event_counts[event_value] += 1
                    previous_value = event_value

                sample_index += 1

    return {
        "status_channel_name": STATUS_CHANNEL_NAME,
        "status_channel_index": status_index,
        "sampling_rate": sampling_rate,
        "event_value_counts": dict(sorted(event_value_counts.items())),
        "rising_event_counts": dict(sorted(rising_event_counts.items())),
        "events": events,
    }


def infer_trial_boundaries(events: list[dict]) -> list[dict]:
    """Infer DEAP trial boundaries from status events 3, 4, and 5."""
    baseline_starts = [
        event for event in events if event["event_code"] == EVENT_BASELINE_START
    ]
    stimulus_starts = [
        event for event in events if event["event_code"] == EVENT_STIMULUS_START
    ]
    stimulus_ends = [
        event for event in events if event["event_code"] == EVENT_STIMULUS_END
    ]

    trial_count = min(len(baseline_starts), len(stimulus_starts), len(stimulus_ends))
    trials = []

    for trial_index in range(trial_count):
        baseline_start = baseline_starts[trial_index]
        stimulus_start = stimulus_starts[trial_index]
        stimulus_end = stimulus_ends[trial_index]

        trials.append(
            {
                "trial": trial_index + 1,
                "baseline_start_sample": baseline_start["sample"],
                "stimulus_start_sample": stimulus_start["sample"],
                "stimulus_end_sample": stimulus_end["sample"],
                "baseline_start_time": baseline_start["time_seconds"],
                "stimulus_start_time": stimulus_start["time_seconds"],
                "stimulus_end_time": stimulus_end["time_seconds"],
                "baseline_duration_seconds": (
                    stimulus_start["time_seconds"] - baseline_start["time_seconds"]
                ),
                "stimulus_duration_seconds": (
                    stimulus_end["time_seconds"] - stimulus_start["time_seconds"]
                ),
            }
        )

    return trials


def get_trial_boundaries_from_bdf(bdf_path: str | Path) -> list[dict]:
    """Return inferred trial time boundaries for one raw DEAP BDF file.

    The returned list contains timing and sample-index boundaries only. It does
    not load EEG channels, save trial data, or apply preprocessing.
    """
    event_info = extract_status_events(bdf_path)
    return infer_trial_boundaries(event_info["events"])


def _read_eeg_interval(
    bdf_path: str | Path,
    header: dict,
    start_sample: int,
    end_sample: int,
    eeg_channel_count: int = EEG_CHANNEL_COUNT,
) -> list[array]:
    """Read raw digital EEG samples for one interval and first EEG channels."""
    path = Path(bdf_path)
    samples_per_record = header["samples_per_record"]
    eeg_samples_per_record = samples_per_record[0]

    for channel_index in range(eeg_channel_count):
        if samples_per_record[channel_index] != eeg_samples_per_record:
            raise ValueError("EEG channels have inconsistent samples per record.")

    first_record = start_sample // eeg_samples_per_record
    last_record = (end_sample - 1) // eeg_samples_per_record
    record_bytes = sum(samples_per_record) * 3
    eeg_channel_bytes = eeg_samples_per_record * 3
    channel_data = [array("i") for _ in range(eeg_channel_count)]

    with path.open("rb") as file:
        for record_index in range(first_record, last_record + 1):
            record_start_sample = record_index * eeg_samples_per_record
            local_start = max(start_sample - record_start_sample, 0)
            local_end = min(end_sample - record_start_sample, eeg_samples_per_record)

            for channel_index in range(eeg_channel_count):
                channel_offset = channel_index * eeg_channel_bytes
                file.seek(
                    header["header_bytes"]
                    + record_index * record_bytes
                    + channel_offset
                )
                raw_channel = file.read(eeg_channel_bytes)
                _append_signed_int24_samples(
                    channel_data[channel_index],
                    raw_channel,
                    local_start,
                    local_end,
                )

    return channel_data


def extract_raw_eeg_trials_from_bdf(
    bdf_path: str | Path,
    eeg_channel_count: int = EEG_CHANNEL_COUNT,
) -> dict:
    """Cut raw EEG trial snippets from one subject's continuous BDF file.

    This function only extracts raw digital samples. It does not filter,
    remove artifacts, baseline-correct, extract features, or save files.
    """
    path = Path(bdf_path)
    header = read_bdf_header(path)
    boundaries = get_trial_boundaries_from_bdf(path)
    sampling_rate = header["sampling_rates"][0]
    trials = []

    for boundary in boundaries:
        baseline = _read_eeg_interval(
            path,
            header,
            boundary["baseline_start_sample"],
            boundary["stimulus_start_sample"],
            eeg_channel_count,
        )
        stimulus = _read_eeg_interval(
            path,
            header,
            boundary["stimulus_start_sample"],
            boundary["stimulus_end_sample"],
            eeg_channel_count,
        )
        trials.append(
            {
                "trial": boundary["trial"],
                "boundary": boundary,
                "baseline": baseline,
                "stimulus": stimulus,
            }
        )

    return {
        "file_name": path.name,
        "sampling_rate": sampling_rate,
        "eeg_channel_labels": header["channel_labels"][:eeg_channel_count],
        "trials": trials,
    }


def summarize_extracted_raw_eeg_trials(extracted: dict) -> dict:
    """Summarize shapes and memory use of extracted raw EEG trial snippets."""
    trials = extracted["trials"]
    first_trial = trials[0]
    baseline_sample_counts = [
        len(trial["baseline"][0]) for trial in trials
    ]
    stimulus_sample_counts = [
        len(trial["stimulus"][0]) for trial in trials
    ]
    baseline_channel_count = len(first_trial["baseline"])
    stimulus_channel_count = len(first_trial["stimulus"])
    baseline_total_values = sum(
        len(channel) for trial in trials for channel in trial["baseline"]
    )
    stimulus_total_values = sum(
        len(channel) for trial in trials for channel in trial["stimulus"]
    )
    bytes_per_sample = first_trial["baseline"][0].itemsize

    return {
        "file_name": extracted["file_name"],
        "trial_count": len(trials),
        "eeg_channel_count": baseline_channel_count,
        "baseline_sample_counts": baseline_sample_counts,
        "stimulus_sample_counts": stimulus_sample_counts,
        "baseline_shape_first_trial": (
            baseline_channel_count,
            len(first_trial["baseline"][0]),
        ),
        "stimulus_shape_first_trial": (
            stimulus_channel_count,
            len(first_trial["stimulus"][0]),
        ),
        "all_baseline_shape": (
            len(trials),
            baseline_channel_count,
            "variable_samples",
        ),
        "all_stimulus_shape": (
            len(trials),
            stimulus_channel_count,
            "variable_samples",
        ),
        "stored_sample_type": "array('i') signed 24-bit raw digital values",
        "estimated_memory_mb": (
            (baseline_total_values + stimulus_total_values)
            * bytes_per_sample
            / (1024 * 1024)
        ),
    }


def _crop_or_pad_channel(
    channel: array,
    target_samples: int,
    trial_index: int,
    segment_name: str,
    channel_index: int,
    warnings: list[str],
) -> array:
    """Crop one channel to fixed length or pad short data with zeros."""
    current_samples = len(channel)

    if current_samples >= target_samples:
        return array("i", channel[:target_samples])

    warning = (
        f"trial {trial_index}, {segment_name}, channel {channel_index + 1}: "
        f"{current_samples} samples < target {target_samples}; "
        "zero-padding is applied to keep a fixed shape."
    )
    warnings.append(warning)

    fixed_channel = array("i", channel)
    fixed_channel.extend([0] * (target_samples - current_samples))
    return fixed_channel


def standardize_raw_eeg_trial_lengths(
    extracted: dict,
    baseline_samples: int = TARGET_BASELINE_SAMPLES,
    stimulus_samples: int = TARGET_STIMULUS_SAMPLES,
) -> dict:
    """Crop/pad raw EEG trial snippets to fixed DEAP baseline/stimulus lengths.

    This step only organizes raw data into uniform shapes. It does not filter,
    apply ICA, remove artifacts, baseline-correct, or extract features.
    """
    warnings = []
    standardized_trials = []

    for trial in extracted["trials"]:
        fixed_baseline = []
        fixed_stimulus = []

        for channel_index, channel in enumerate(trial["baseline"]):
            fixed_baseline.append(
                _crop_or_pad_channel(
                    channel,
                    baseline_samples,
                    trial["trial"],
                    "baseline",
                    channel_index,
                    warnings,
                )
            )

        for channel_index, channel in enumerate(trial["stimulus"]):
            fixed_stimulus.append(
                _crop_or_pad_channel(
                    channel,
                    stimulus_samples,
                    trial["trial"],
                    "stimulus",
                    channel_index,
                    warnings,
                )
            )

        standardized_trials.append(
            {
                "trial": trial["trial"],
                "boundary": trial["boundary"],
                "baseline": fixed_baseline,
                "stimulus": fixed_stimulus,
            }
        )

    trial_count = len(standardized_trials)
    channel_count = len(standardized_trials[0]["baseline"]) if trial_count else 0

    return {
        "file_name": extracted["file_name"],
        "sampling_rate": extracted["sampling_rate"],
        "eeg_channel_labels": extracted["eeg_channel_labels"],
        "trials": standardized_trials,
        "baseline_shape": (trial_count, channel_count, baseline_samples),
        "stimulus_shape": (trial_count, channel_count, stimulus_samples),
        "warnings": warnings,
    }


def _require_signal_processing_dependencies():
    """Import NumPy/SciPy only when the preprocessing chain is requested."""
    try:
        import numpy as np
        from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "Basic preprocessing requires numpy and scipy. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    return np, butter, filtfilt, iirnotch, sosfiltfilt


def _segments_to_numpy(trials: list[dict], segment_name: str):
    """Convert fixed-length array('i') trial segments to a 3D NumPy array."""
    np, _, _, _, _ = _require_signal_processing_dependencies()
    return np.asarray(
        [
            [list(channel) for channel in trial[segment_name]]
            for trial in trials
        ],
        dtype=np.float32,
    )


def bandpass_filter_eeg(
    eeg_data,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
):
    """Apply Butterworth bandpass filtering to EEG data.

    Expected input shape: (trials, channels, samples). Filtering is applied
    along the last axis. No ICA, artifact removal, or feature extraction is
    performed here.
    """
    _, butter, _, _, sosfiltfilt = _require_signal_processing_dependencies()
    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return sosfiltfilt(sos, eeg_data, axis=-1).astype("float32")


def notch_filter_eeg(
    eeg_data,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    notch_hz: float = NOTCH_FREQ_HZ,
    quality_factor: float = NOTCH_QUALITY_FACTOR,
):
    """Apply a 50 Hz notch filter to suppress mains interference."""
    _, _, filtfilt, iirnotch, _ = _require_signal_processing_dependencies()
    b, a = iirnotch(notch_hz, quality_factor, fs=sampling_rate)
    return filtfilt(b, a, eeg_data, axis=-1).astype("float32")


def baseline_correct_stimulus(filtered_baseline, filtered_stimulus):
    """Subtract each trial/channel baseline mean from the stimulus segment."""
    baseline_mean = filtered_baseline.mean(axis=-1, keepdims=True)
    return (filtered_stimulus - baseline_mean).astype("float32")


def run_basic_preprocessing_on_standardized_trials(standardized: dict) -> dict:
    """Run the minimal baseline preprocessing chain on fixed-length raw trials.

    Steps:
    1. Convert fixed-length raw digital samples to NumPy arrays.
    2. Apply 4-45 Hz bandpass filtering to baseline and stimulus.
    3. Apply 50 Hz notch filtering to the bandpass-filtered data.
    4. Correct stimulus by subtracting the filtered baseline mean.
    """
    raw_baseline = _segments_to_numpy(standardized["trials"], "baseline")
    raw_stimulus = _segments_to_numpy(standardized["trials"], "stimulus")
    sampling_rate = standardized["sampling_rate"]

    bandpassed_baseline = bandpass_filter_eeg(raw_baseline, sampling_rate)
    bandpassed_stimulus = bandpass_filter_eeg(raw_stimulus, sampling_rate)
    filtered_baseline = notch_filter_eeg(bandpassed_baseline, sampling_rate)
    filtered_stimulus = notch_filter_eeg(bandpassed_stimulus, sampling_rate)
    corrected_stimulus = baseline_correct_stimulus(
        filtered_baseline,
        filtered_stimulus,
    )

    return {
        "file_name": standardized["file_name"],
        "sampling_rate": sampling_rate,
        "filter_parameters": {
            "bandpass_hz": (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ),
            "bandpass_order": BANDPASS_ORDER,
            "notch_hz": NOTCH_FREQ_HZ,
            "notch_quality_factor": NOTCH_QUALITY_FACTOR,
        },
        "raw_fixed_baseline": raw_baseline,
        "raw_fixed_stimulus": raw_stimulus,
        "filtered_baseline": filtered_baseline,
        "filtered_stimulus": filtered_stimulus,
        "baseline_corrected_stimulus": corrected_stimulus,
    }


def summarize_subject_trial_boundaries(subject_id: int) -> dict:
    """Summarize Status-code counts and first trial durations for one subject."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    event_info = extract_status_events(bdf_path)
    trials = infer_trial_boundaries(event_info["events"])
    counts = event_info["rising_event_counts"]

    return {
        "subject_id": subject_id,
        "file_name": bdf_path.name,
        "code_3_count": counts.get(EVENT_BASELINE_START, 0),
        "code_4_count": counts.get(EVENT_STIMULUS_START, 0),
        "code_5_count": counts.get(EVENT_STIMULUS_END, 0),
        "inferred_trial_count": len(trials),
        "first_3_baseline_durations": [
            trial["baseline_duration_seconds"] for trial in trials[:3]
        ],
        "first_3_stimulus_durations": [
            trial["stimulus_duration_seconds"] for trial in trials[:3]
        ],
        "is_consistent": (
            counts.get(EVENT_BASELINE_START, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and counts.get(EVENT_STIMULUS_START, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and counts.get(EVENT_STIMULUS_END, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and len(trials) == EXPECTED_DEAP_TRIAL_COUNT
        ),
    }


def validate_multi_subject_trial_boundaries(
    subject_ids: tuple[int, ...] = (1, 2, 3),
) -> list[dict]:
    """Validate trial-boundary consistency across multiple raw BDF files."""
    return [
        summarize_subject_trial_boundaries(subject_id)
        for subject_id in subject_ids
    ]


def load_participant_ratings(
    participant_id: int,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
) -> list[dict]:
    """Load participant ratings for one DEAP subject from metadata CSV."""
    metadata_path = PROJECT_ROOT / Path(metadata_dir) / "participant_ratings.csv"
    participant_rows = []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row["Participant_id"]) == participant_id:
                participant_rows.append(row)

    return participant_rows


def inspect_subject_raw(
    subject_id: int = 1,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
) -> dict:
    """Inspect one subject's raw BDF file and related label metadata."""
    bdf_path = PROJECT_ROOT / Path(original_dir) / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    ratings = load_participant_ratings(subject_id, metadata_dir)

    label_columns = ["Valence", "Arousal", "Dominance", "Liking", "Familiarity"]
    sampling_rate = header["sampling_rates"][0]
    expected_trial_samples = int(DEAP_TRIAL_DURATION_SECONDS * sampling_rate)

    return {
        **header,
        "trial_count_from_metadata": len(ratings),
        "eeg_channel_count_expected": EEG_CHANNEL_COUNT,
        "label_columns": label_columns,
        "label_shape": (len(ratings), len(label_columns)),
        "expected_trial_duration_seconds": DEAP_TRIAL_DURATION_SECONDS,
        "expected_trial_samples_per_channel": expected_trial_samples,
    }


def print_subject_raw_report(subject_id: int = 1) -> None:
    """Print a compact report for one subject's raw DEAP data structure."""
    info = inspect_subject_raw(subject_id)

    print("DEAP raw data structure check")
    print(f"file name: {info['file_name']}")
    print(f"data type: {info['data_type']}")
    print(
        "data dimension: "
        f"{info['num_signals']} channels x "
        f"{info['total_samples_per_channel']} continuous samples"
    )
    print(f"trial count: {info['trial_count_from_metadata']} metadata trials")
    print(
        "channel count: "
        f"{info['num_signals']} raw channels "
        f"({info['eeg_channel_count_expected']} EEG channels expected by DEAP)"
    )
    print(
        "per-trial duration/sample count: "
        f"{info['expected_trial_duration_seconds']} s, "
        f"{info['expected_trial_samples_per_channel']} samples/channel "
        "(expected stimulus segment; raw BDF is continuous and not segmented here)"
    )
    print(f"label information dimension: {info['label_shape']}")
    print(f"label columns: {', '.join(info['label_columns'])}")
    print(f"first 10 channel labels: {', '.join(info['channel_labels'][:10])}")


def print_subject_event_report(subject_id: int = 1) -> None:
    """Print event-channel and trial-boundary checks for one raw subject."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    event_info = extract_status_events(bdf_path)
    trials = infer_trial_boundaries(event_info["events"])

    print("DEAP raw BDF event structure check")
    print(f"file name: {bdf_path.name}")
    print(
        "event channel: "
        f"{event_info['status_channel_name']} "
        f"(index {event_info['status_channel_index']})"
    )
    print(f"sampling rate: {event_info['sampling_rate']} Hz")
    print(f"extracted rising events: {len(event_info['events'])}")
    print(f"event code counts: {event_info['rising_event_counts']}")
    print("first 20 rising events:")

    for event in event_info["events"][:20]:
        print(
            f"  sample={event['sample']}, "
            f"time={event['time_seconds']:.3f}s, "
            f"code={event['event_code']}"
        )

    print("trial-boundary inference:")
    print(
        f"  code {EVENT_BASELINE_START}: baseline start, "
        f"count={event_info['rising_event_counts'].get(EVENT_BASELINE_START, 0)}"
    )
    print(
        f"  code {EVENT_STIMULUS_START}: stimulus start, "
        f"count={event_info['rising_event_counts'].get(EVENT_STIMULUS_START, 0)}"
    )
    print(
        f"  code {EVENT_STIMULUS_END}: stimulus end, "
        f"count={event_info['rising_event_counts'].get(EVENT_STIMULUS_END, 0)}"
    )
    print(f"  inferred trials: {len(trials)}")
    print("first 5 inferred trials:")

    for trial in trials[:5]:
        print(
            f"  trial={trial['trial']}, "
            f"baseline_start={trial['baseline_start_time']:.3f}s, "
            f"stimulus_start={trial['stimulus_start_time']:.3f}s, "
            f"stimulus_end={trial['stimulus_end_time']:.3f}s, "
            f"baseline={trial['baseline_duration_seconds']:.3f}s, "
            f"stimulus={trial['stimulus_duration_seconds']:.3f}s"
        )

    print("segmentation plan:")
    print(
        "  Each trial is inferred as [code 3, code 5), containing a baseline "
        "segment [code 3, code 4) and a stimulus segment [code 4, code 5)."
    )
    print(
        f"  Expected durations are about {DEAP_BASELINE_DURATION_SECONDS}s "
        f"baseline and {DEAP_TRIAL_DURATION_SECONDS}s stimulus."
    )
    print(
        "  This script only verifies the boundary logic and does not save "
        "segmented trials."
    )
    print("uncertainty:")
    print(
        "  The event-code meaning is inferred from repeated counts and timing "
        "patterns, not from an external marker manual in this repository."
    )


def print_multi_subject_consistency_report(
    subject_ids: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Print Status-event consistency checks for multiple subjects."""
    summaries = validate_multi_subject_trial_boundaries(subject_ids)
    all_consistent = all(summary["is_consistent"] for summary in summaries)

    print("DEAP multi-subject trial-boundary consistency check")
    print("verified facts:")

    for summary in summaries:
        baseline_durations = ", ".join(
            f"{duration:.3f}s"
            for duration in summary["first_3_baseline_durations"]
        )
        stimulus_durations = ", ".join(
            f"{duration:.3f}s"
            for duration in summary["first_3_stimulus_durations"]
        )

        print(f"  subject s{summary['subject_id']:02d}: {summary['file_name']}")
        print(
            "    code counts: "
            f"3={summary['code_3_count']}, "
            f"4={summary['code_4_count']}, "
            f"5={summary['code_5_count']}"
        )
        print(f"    inferred trials: {summary['inferred_trial_count']}")
        print(f"    first 3 baseline durations: {baseline_durations}")
        print(f"    first 3 stimulus durations: {stimulus_durations}")

    print("conclusion based on verified facts:")

    if all_consistent:
        print(
            "  For s01, s02, and s03, event codes 3, 4, and 5 each appear "
            "40 times, so the same trial-boundary rule is consistent across "
            "these checked subjects."
        )
        print(
            "  The fixed rule is: trial=[code 3, code 5), "
            "baseline=[code 3, code 4), stimulus=[code 4, code 5)."
        )
        print(
            "  The function get_trial_boundaries_from_bdf(bdf_path) now "
            "returns the 40 boundary records for one subject without saving "
            "EEG segments."
        )
    else:
        print(
            "  The checked subjects are not fully consistent. Trial splitting "
            "should not be finalized until the mismatched file is inspected."
        )


def print_raw_eeg_trial_extraction_report(subject_id: int = 1) -> None:
    """Cut one subject into raw EEG trial snippets and print shape details."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    summary = summarize_extracted_raw_eeg_trials(extracted)

    baseline_counts_preview = ", ".join(
        str(count) for count in summary["baseline_sample_counts"][:5]
    )
    stimulus_counts_preview = ", ".join(
        str(count) for count in summary["stimulus_sample_counts"][:5]
    )

    print("DEAP raw EEG trial extraction check")
    print(f"file name: {summary['file_name']}")
    print(f"trial count: {summary['trial_count']}")
    print(f"EEG channel count: {summary['eeg_channel_count']}")
    print(f"baseline samples per trial, first 5: {baseline_counts_preview}")
    print(f"stimulus samples per trial, first 5: {stimulus_counts_preview}")
    print(f"baseline array shape, first trial: {summary['baseline_shape_first_trial']}")
    print(f"stimulus array shape, first trial: {summary['stimulus_shape_first_trial']}")
    print(f"all baseline shape: {summary['all_baseline_shape']}")
    print(f"all stimulus shape: {summary['all_stimulus_shape']}")
    print(f"stored sample type: {summary['stored_sample_type']}")
    print(f"estimated in-memory data size: {summary['estimated_memory_mb']:.1f} MB")
    print("implementation note:")
    print(
        "  The current implementation reads only the first 32 EEG channels and "
        "only the verified trial intervals. It does not load peripheral "
        "channels or the Status channel into the extracted trial data."
    )
    print(
        "  The sample counts can vary slightly because boundaries come from "
        "real event sample indices rather than forced 5s/60s rounding."
    )


def print_standardized_trial_length_report(subject_id: int = 1) -> None:
    """Print fixed-length raw EEG trial organization details."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    standardized = standardize_raw_eeg_trial_lengths(extracted)

    print("DEAP fixed-length raw EEG trial organization check")
    print(f"file name: {standardized['file_name']}")
    print(f"target baseline length: {TARGET_BASELINE_SAMPLES} samples")
    print(f"target stimulus length: {TARGET_STIMULUS_SAMPLES} samples")
    print(f"baseline shape: {standardized['baseline_shape']}")
    print(f"stimulus shape: {standardized['stimulus_shape']}")

    if standardized["warnings"]:
        print("warnings:")
        for warning in standardized["warnings"]:
            print(f"  {warning}")
    else:
        print(
            "warnings: none; all trial segments were at least the target "
            "length and were cropped to the fixed shape where needed."
        )

    print("processing note:")
    print(
        "  This step only enforces fixed lengths: 5s baseline and 60s "
        "stimulus at 512 Hz. No filtering, ICA, artifact removal, baseline "
        "correction, feature extraction, classification, or active learning "
        "is performed."
    )


def print_basic_preprocessing_report(subject_id: int = 1) -> None:
    """Run and print the minimal basic preprocessing-chain verification."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    standardized = standardize_raw_eeg_trial_lengths(extracted)
    preprocessed = run_basic_preprocessing_on_standardized_trials(standardized)
    params = preprocessed["filter_parameters"]

    print("DEAP basic EEG preprocessing chain check")
    print(f"file name: {preprocessed['file_name']}")
    print(
        "filter parameters: "
        f"bandpass={params['bandpass_hz'][0]}-{params['bandpass_hz'][1]} Hz, "
        f"order={params['bandpass_order']}; "
        f"notch={params['notch_hz']} Hz, Q={params['notch_quality_factor']}"
    )
    print("raw fixed-length data:")
    print(f"  baseline shape: {preprocessed['raw_fixed_baseline'].shape}")
    print(f"  stimulus shape: {preprocessed['raw_fixed_stimulus'].shape}")
    print("filtered data after bandpass + notch:")
    print(f"  baseline shape: {preprocessed['filtered_baseline'].shape}")
    print(f"  stimulus shape: {preprocessed['filtered_stimulus'].shape}")
    print("baseline-corrected stimulus data:")
    print(
        "  stimulus shape: "
        f"{preprocessed['baseline_corrected_stimulus'].shape}"
    )
    print("processing note:")
    print(
        "  Baseline correction subtracts each trial/channel's filtered 5s "
        "baseline mean from the corresponding 60s filtered stimulus."
    )
    print(
        "  This report only verifies that the basic preprocessing chain runs. "
        "No ICA, feature extraction, classification, or active learning is "
        "performed."
    )


def main() -> None:
    print_subject_raw_report(subject_id=1)
    print()
    print_subject_event_report(subject_id=1)
    print()
    print_multi_subject_consistency_report(subject_ids=(1, 2, 3))
    print()
    print_raw_eeg_trial_extraction_report(subject_id=1)
    print()
    print_standardized_trial_length_report(subject_id=1)
    print()
    print_basic_preprocessing_report(subject_id=1)


if __name__ == "__main__":
    main()
