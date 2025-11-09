import numpy as np
import pandas as pd

def welch_t_stat(control, treatment):
    """
    Calculate Welch's t-statistic.

    Calculates the t-statistic of the difference in means between two groups
    allowing for each group to have its own variance.

    Parameters
    ----------
    control : array-like
      Observed outcomes of the control group.
    treatment : array-like
      Observed outcomes of the treatment group.

    Returns
    ----------
    float
      Welch's observed t-statistic.

    Examples
    ----------
    ```
    # Calculate difference of means between two groups
    welch_t_stat(
        [1,2,3,4,5],
        [9,8,0,7,6,5,6]
    )
    ```
    """
    # Quality Assurance (QA)
    valid_types = (list, tuple, np.ndarray, pd.Series)
    if not isinstance(control, valid_types):
        raise TypeError(
            "Control must be an array-like object, including lists, tuples, "
            "NumPy arrays, or Pandas Series."
        )

    if not isinstance(treatment, valid_types):
        raise TypeError(
            "Treatment must be a list, tuple, NumPy array, or Pandas Series."
        )

    # Calculate means
    mean_c = np.mean(control)
    mean_t = np.mean(treatment)

    # Calculate standart deviation under H_0
    sdev = np.sqrt(
        np.var(control, ddof=1) / len(control)
        + np.var(treatment, ddof=1) / len(treatment))

    # Calculate statistic
    t_stat = (mean_t - mean_c) / sdev

    # Return
    return float(t_stat)
