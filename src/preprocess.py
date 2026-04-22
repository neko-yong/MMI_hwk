"""Minimal EEG preprocessing utilities.

This module will later contain the preprocessing pipeline for the DEAP dataset.
For now, it only provides a very small placeholder API so the project structure
is ready for implementation.
"""

from __future__ import annotations

import numpy as np


def preprocess_eeg(eeg_data: np.ndarray) -> np.ndarray:
    """Return EEG data without modification as a placeholder step."""
    return eeg_data
