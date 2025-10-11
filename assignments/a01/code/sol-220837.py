import numpy as np
import pandas as pd
import os

def welch_t_stat(control, treatment):
    """
    Compute Welch's t-statistic for the difference in means between two samples.

    Parameters
    ----------
    control : array-like
        Observations for the control group.
    treatment : array-like
        Observations for the treatment group.

    Returns
    -------
    float
        The Welch t-statistic.
    """

    valid_types = (list, tuple, np.ndarray, pd.Series)
    if not isinstance(control, valid_types):
        raise TypeError("Argument 'control' must be list, tuple, numpy.ndarray, or pandas.Series.")
    if not isinstance(treatment, valid_types):
        raise TypeError("Argument 'treatment' must be list, tuple, numpy.ndarray, or pandas.Series.")

    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    assert control.ndim == 1, "Argument 'control' must be one-dimensional."
    assert treatment.ndim == 1, "Argument 'treatment' must be one-dimensional."
    assert len(control) > 1 and len(treatment) > 1, "Both samples must have at least two observations."

    mean_c = np.mean(control)
    mean_t = np.mean(treatment)
    var_c = np.var(control, ddof=1)
    var_t = np.var(treatment, ddof=1)
    n_c = len(control)
    n_t = len(treatment)

    denom = np.sqrt(var_t / n_t + var_c / n_c)
    if denom == 0:
        raise ValueError("The pooled standard error is zero; t-statistic undefined.")

    return float((mean_t - mean_c) / denom)


if __name__ == "__main__":
    # Load data from the relative path
    data_path = os.path.join("assignments", "a01", "data", "data.csv")
    df = pd.read_csv(data_path)

    # Expecting two columns: ['value', 'treatment']
    if df.shape[1] != 2:
        raise ValueError("Expected data.csv with two columns: values and treatment indicator.")

    value_col = df.columns[0]
    treat_col = df.columns[1]

    control = df.loc[df[treat_col] == 0, value_col]
    treatment = df.loc[df[treat_col] == 1, value_col]

    t_stat = welch_t_stat(control, treatment)
    print("Welch t-statistic:", t_stat)
