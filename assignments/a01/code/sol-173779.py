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

    Examples
    --------
    ```
    # Calculate t-stat for two groups
    welch_t_stat(
        [1, 2, 3, 4, 5],
        [9, 8, 0, 7, 6, 5, 6]
    )
    ```
    """

    # Quality Assurance (QA)
    TYPES = (list, tuple, np.ndarray, pd.Series)
    if not isinstance(control, TYPES):
        raise TypeError(
            'Incorrect data type for argument `control`. Expected list, tuple,'
            f' numpy.ndarray or pandas.Series.\nReceived {type(control)} instead.'
        )
    if not isinstance(treatment, TYPES):
        raise TypeError(
            'Incorrect data type for argument `treatment`. Expected list, tuple,'
            f' numpy.ndarray or pandas.Series.\nReceived {type(treatment)} instead.'
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
