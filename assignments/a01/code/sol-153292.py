import numpy as np

# Define the acceptable array-like types for error checking
ARRAY_LIKE_TYPES = (list, tuple, np.ndarray)

def welch_t_stat(control, treatment):
    # Quality assurance (QA)

    # 1. Type Guardrail (Required by Assignment)
    if not isinstance(control, ARRAY_LIKE_TYPES):
        raise TypeError(f"Expected 'control' to be an array-like object, but got {type(control).__name__}")
    if not isinstance(treatment, ARRAY_LIKE_TYPES):
        raise TypeError(f"Expected 'treatment' to be an array-like object, but got {type(treatment).__name__}")

    # Convert to NumPy arrays for calculation consistency and handling array-like inputs
    control_arr = np.array(control)
    treatment_arr = np.array(treatment)

    # 2. Size Guardrail (Assertion for non-empty arrays)
    if control_arr.size == 0 or treatment_arr.size == 0:
        raise AssertionError("Both 'control' and 'treatment' groups must be non-empty.")

    # Calculate means
    # NOTE: Using the numpy array versions ensures the mean function is available.
    mean_c = np.mean(control_arr)
    mean_t = np.mean(treatment_arr)

    # Calculate standard deviation under H_0 (Standard Error of the Difference)
    # The assignment requires using the provided calculation structure.
    sdev = np.sqrt(
         np.var(control_arr, ddof=1) / len(control_arr)
         + np.var(treatment_arr, ddof=1) / len(treatment_arr)
    )

    # Calculate statistic
    t_stat = (mean_t - mean_c) / sdev

    return float(t_stat)


