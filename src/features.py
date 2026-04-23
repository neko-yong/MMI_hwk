"""Minimal EEG feature extraction utilities.

Features in this module are intended to be extracted from preprocessed DEAP
stimulus data, specifically the baseline-corrected 60-second stimulus segment.

Expected input shape:
    (n_trials, n_channels, n_samples)

For the current project setup this is usually:
    (40, 32, 30720)

Each trial is converted into one feature vector. Features are aggregated per
channel over the time axis and then concatenated across channels.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


DEFAULT_SAMPLING_RATE = 512
DEFAULT_FREQUENCY_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _validate_stimulus_data(stimulus_data: np.ndarray) -> np.ndarray:
    """Validate and convert stimulus data to a floating-point 3D array."""
    data = np.asarray(stimulus_data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            "Expected stimulus data with shape "
            "(n_trials, n_channels, n_samples)."
        )

    return data


def extract_time_domain_features(stimulus_data: np.ndarray) -> np.ndarray:
    """Extract simple time-domain features from stimulus EEG data.

    The features are computed for each trial and channel over the sample axis:
    mean, variance, and standard deviation.

    Output shape:
        (n_trials, n_channels * 3)
    """
    data = _validate_stimulus_data(stimulus_data)

    mean = np.mean(data, axis=-1)
    variance = np.var(data, axis=-1)
    standard_deviation = np.std(data, axis=-1)

    return np.concatenate(
        [mean, variance, standard_deviation],
        axis=1,
    )


def compute_band_power(
    stimulus_data: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int = 1024,
) -> np.ndarray:
    """Compute theta/alpha/beta/gamma band power for each trial and channel.

    PSD is estimated with Welch's method along the time axis. For each frequency
    band, PSD values inside the band are integrated with the trapezoidal rule.

    Output shape:
        (n_trials, n_channels, n_bands)
    """
    data = _validate_stimulus_data(stimulus_data)
    bands = frequency_bands or DEFAULT_FREQUENCY_BANDS
    segment_length = min(nperseg, data.shape[-1])

    frequencies, psd = welch(
        data,
        fs=sampling_rate,
        nperseg=segment_length,
        axis=-1,
    )

    band_powers = []

    for low_hz, high_hz in bands.values():
        band_mask = (frequencies >= low_hz) & (frequencies < high_hz)

        if not np.any(band_mask):
            raise ValueError(
                f"No PSD frequency bins found for band {low_hz}-{high_hz} Hz."
            )

        power = np.trapezoid(
            psd[..., band_mask],
            frequencies[band_mask],
            axis=-1,
        )
        band_powers.append(power)

    return np.stack(band_powers, axis=-1)


def extract_frequency_domain_features(
    stimulus_data: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Extract common EEG band-power features from stimulus data.

    The current frequency-domain features are absolute band powers for:
    theta, alpha, beta, and gamma.

    Output shape:
        (n_trials, n_channels * 4)
    """
    band_power = compute_band_power(
        stimulus_data,
        sampling_rate=sampling_rate,
        frequency_bands=frequency_bands,
    )
    return band_power.reshape(band_power.shape[0], -1)


def extract_features(
    stimulus_data: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Extract the minimal feature matrix from preprocessed stimulus data.

    Input should be baseline-corrected stimulus data, not baseline data and not
    unsegmented continuous EEG.

    Feature layout per channel:
    - time domain: mean, variance, standard deviation
    - frequency domain: theta, alpha, beta, gamma band power

    With 32 EEG channels, each trial produces:
        32 * (3 + 4) = 224 features

    Output shape:
        X = (n_trials, n_features)
    """
    data = _validate_stimulus_data(stimulus_data)
    time_features = extract_time_domain_features(data)
    frequency_features = extract_frequency_domain_features(
        data,
        sampling_rate=sampling_rate,
        frequency_bands=frequency_bands,
    )

    return np.concatenate(
        [time_features, frequency_features],
        axis=1,
    )


def describe_feature_matrix(
    stimulus_data: np.ndarray,
    feature_matrix: np.ndarray,
    frequency_bands: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Return a small summary explaining the extracted feature matrix."""
    data = _validate_stimulus_data(stimulus_data)
    bands = frequency_bands or DEFAULT_FREQUENCY_BANDS
    features_per_channel = 3 + len(bands)

    return {
        "input_segment": "baseline-corrected stimulus data",
        "input_shape": data.shape,
        "time_features": ["mean", "variance", "standard_deviation"],
        "frequency_features": list(bands.keys()),
        "aggregation": "features are computed per trial and channel over time",
        "features_per_channel": features_per_channel,
        "features_per_trial": data.shape[1] * features_per_channel,
        "feature_matrix_shape": feature_matrix.shape,
    }


def print_feature_extraction_report(
    stimulus_data: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
) -> np.ndarray:
    """Extract features and print a compact shape report."""
    feature_matrix = extract_features(
        stimulus_data,
        sampling_rate=sampling_rate,
    )
    summary = describe_feature_matrix(stimulus_data, feature_matrix)

    print("EEG feature extraction check")
    print(f"input segment: {summary['input_segment']}")
    print(f"input shape: {summary['input_shape']}")
    print(f"time features: {summary['time_features']}")
    print(f"frequency features: {summary['frequency_features']}")
    print(f"aggregation: {summary['aggregation']}")
    print(f"features per channel: {summary['features_per_channel']}")
    print(f"features per trial: {summary['features_per_trial']}")
    print(f"feature matrix X shape: {summary['feature_matrix_shape']}")

    return feature_matrix


def main() -> None:
    """Run a tiny synthetic smoke test for the feature module."""
    rng = np.random.default_rng(42)
    demo_stimulus = rng.normal(size=(2, 32, 1024)).astype(np.float32)
    print_feature_extraction_report(demo_stimulus)


if __name__ == "__main__":
    main()
