import numpy as np

def welch_t_stat(control, treatment):
    """
    Calculate the Welch's t-statistic for two independent samples.
    Accepts lists, tuples, numpy arrays, or pandas Series.
    """
    # Convert inputs to numpy arrays
    control = np.asarray(control)
    treatment = np.asarray(treatment)

    # Validate input types
    if not np.issubdtype(control.dtype, np.number) or not np.issubdtype(treatment.dtype, np.number):
        raise TypeError("Both control and treatment must contain numeric data.")

    # Calculate means and variances
    mean_control, mean_treatment = np.mean(control), np.mean(treatment)
    var_control, var_treatment = np.var(control, ddof=1), np.var(treatment, ddof=1)
    n_control, n_treatment = len(control), len(treatment)

    # Welch's t-statistic
    numerator = mean_treatment - mean_control
    denominator = np.sqrt(var_treatment / n_treatment + var_control / n_control)
    t_stat = numerator / denominator

    return t_stat

