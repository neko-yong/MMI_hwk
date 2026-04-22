"""Minimal EEG feature extraction utilities."""

from __future__ import annotations

import numpy as np


def extract_features(eeg_data: np.ndarray) -> np.ndarray:
    """Return a simple flattened feature vector placeholder."""
    if eeg_data.ndim == 1:
        return eeg_data
    return eeg_data.reshape(eeg_data.shape[0], -1)
