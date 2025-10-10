# sol-189610.py
"""
Solution for Assignment 01: Welch's t-statistic
Author: Diego Macal
"""

import numpy as np
import pandas as pd


def welch_t_stat(control, treatment):
    """
    Compute Welch's t-statistic for two independent samples
    with unequal variances.
    """
    # Validate inputs (must not be strings)
    for name, arr in {"control": control, "treatment": treatment}.items():
        if isinstance(arr, str):
            raise TypeError(f"{name} must be numeric, not string")

    # Convert to numpy arrays
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)

    # Sample sizes
    n_c, n_t = len(c), len(t)

    # Means
    mean_c, mean_t = np.mean(c), np.mean(t)

    # Unbiased variances
    var_c, var_t = np.var(c, ddof=1), np.var(t, ddof=1)

    # Welch's t-statistic
    t_stat = (mean_t - mean_c) / np.sqrt(var_t / n_t + var_c / n_c)
    return t_stat

