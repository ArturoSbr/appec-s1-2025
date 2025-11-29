import numpy as np
import pandas as pd  # Required by the autograder

# Define accepted array-like types
ARRAY_LIKE_TYPES = (list, tuple, np.ndarray, pd.Series)


def welch_t_stat(control, treatment):
    """
    Calculates Welch's t-statistic for the difference in means between two
    samples. The formula is:
    t = (x_bar1 - x_bar0) / sqrt(s1^2/n1 + s0^2/n0)
    """
    # TYPE CHECKS
    if not isinstance(control, ARRAY_LIKE_TYPES):
        raise TypeError(
            "Expected 'control' to be an array-like object "
            "(e.g., list, tuple, numpy.ndarray, pandas.Series), "
            f"but got {type(control).__name__}"
        )

    if not isinstance(treatment, ARRAY_LIKE_TYPES):
        raise TypeError(
            "Expected 'treatment' to be an array-like object "
            "(e.g., list, tuple, numpy.ndarray, pandas.Series), "
            f"but got {type(treatment).__name__}"
        )

    # Convert to NumPy arrays
    control_arr = np.array(control)
    treatment_arr = np.array(treatment)

    # SIZE CHECK (non-empty)
    if control_arr.size == 0 or treatment_arr.size == 0:
        raise AssertionError(
            "Both 'control' and 'treatment' groups must be non-empty."
        )

    # --- Calculation ---
    mean_c = np.mean(control_arr)
    var_c = np.var(control_arr, ddof=1)
    n_c = len(control_arr)

    mean_t = np.mean(treatment_arr)
    var_t = np.var(treatment_arr, ddof=1)
    n_t = len(treatment_arr)

    sdev = np.sqrt((var_t / n_t) + (var_c / n_c))
    t_stat = (mean_t - mean_c) / sdev

    return float(t_stat)