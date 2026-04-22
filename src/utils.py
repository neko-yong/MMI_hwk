"""Shared helper functions for the project."""

from __future__ import annotations

import random

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
