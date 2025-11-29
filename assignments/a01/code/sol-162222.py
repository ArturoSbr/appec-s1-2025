import numpy as np
import pandas as pd

def welch_t_stat(control, treatment):
    """Calculate Welch's t-statistic.

    Calculates the t-statistic of the difference in means between two groups allowing
    for each group to have its own variance.

    Parameters
    ----------
    control : array-like
        Observed outcomes of the control group.
    treatment : array-like
        Observed outcomes of the treatment group.

    Returns
    -------
    float
        Welch's observed t-statistic.
    """

    # 1. TYPE VALIDATION ----------------------------------------------------
    array_like = (list, tuple, np.ndarray, pd.Series)

    if not isinstance(control, array_like):
        raise TypeError("control must be an array-like object.")

    if not isinstance(treatment, array_like):
        raise TypeError("treatment must be an array-like object.")

    # Convert to numpy arrays
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    # 2. GUARDRAILS ---------------------------------------------------------
    assert control.ndim == 1, "control must be one-dimensional."
    assert treatment.ndim == 1, "treatment must be one-dimensional."

    assert len(control) >= 2, "control must contain at least 2 observations."
    assert len(treatment) >= 2, "treatment must contain at least 2 observations."

    assert np.all(np.isfinite(control)), "control contains non-finite values."
    assert np.all(np.isfinite(treatment)), "treatment contains non-finite values."

    # 3. COMPUTE WELCH T-STAT -----------------------------------------------
    mean_diff = treatment.mean() - control.mean()

    var_c = control.var(ddof=1)
    var_t = treatment.var(ddof=1)

    n_c = len(control)
    n_t = len(treatment)

    denom = np.sqrt(var_c / n_c + var_t / n_t)

    return mean_diff / denom
