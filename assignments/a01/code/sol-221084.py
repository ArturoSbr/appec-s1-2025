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

    Example
    -------
    welch_t_stat(
              [1,2,3,4,5],
              [6,7,8,9,10,11]
                )

    """
    # Quality Assurance (QA)
    TYPES = (list, tuple, np.ndarray, pd.Series)
    assert isinstance(control, TYPES), ( #para no repetir el código varias veces
    'Incorrect data type for argument `control`. Expected list, tuple, np.ndarray, pd.Series.'
    f'\nRecieved {type(control)} instead.'
    )
    assert isinstance(treatment, TYPES), ( #para no repetir el código varias veces
    'Incorrect data type for argument `treatment`. Expected list, tuple, np.ndarray, pd.Series.'
    f'\nRecieved {type(treatment)} instead.'
    )

    # Calculate means
    mean_c = np.mean(control)
    mean_t = np.mean(treatment)

    # Calculate standard deviation under H_0
    sdev = np.sqrt(
        np.var(control, ddof=1) / len(control)
        + np.var(treatment, ddof=1) / len(treatment))

    # Calculate statistic
    t_stat = (mean_t - mean_c) / sdev

    # Return
    return float(t_stat)
