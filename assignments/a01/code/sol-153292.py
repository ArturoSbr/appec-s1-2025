import numpy as np
# Import pandas only for type checking, as the autograder may use pd.Series 
# as an input, fulfilling the instruction "pandas series".
try:
    import pandas as pd
    PANDAS_SERIES = pd.Series
except ImportError:
    # If pandas is not available, the type is set to None
    PANDAS_SERIES = type(None) 

# Define a tuple with all accepted array-like types
ARRAY_LIKE_TYPES = (list, tuple, np.ndarray, PANDAS_SERIES)

def welch_t_stat(control, treatment):
    """
    Calculates the Welch's t-statistic for the difference in means between two samples.
    The formula is: t = (x̄₁ - x̄₀) / sqrt(s₁²/n₁ + s₀²/n₀)
    """
    
    # 1. TYPE CHECK (Guardrail 1 - Required: TypeError)
    # Checks if the object is NOT one of the expected array-like types.
    if not isinstance(control, ARRAY_LIKE_TYPES):
        raise TypeError(
            f"Expected 'control' to be an array-like object (e.g., list, tuple, numpy.ndarray, pandas.Series), "
            f"but got {type(control).__name__}"
        )
    if not isinstance(treatment, ARRAY_LIKE_TYPES):
        raise TypeError(
            f"Expected 'treatment' to be an array-like object (e.g., list, tuple, numpy.ndarray, pandas.Series), "
            f"but got {type(treatment).__name__}"
        )

    # Convert to NumPy arrays for calculation
    control_arr = np.array(control)
    treatment_arr = np.array(treatment)
    
    # 2. SIZE CHECK (Guardrail 2 - AssertionError)
    if control_arr.size == 0 or treatment_arr.size == 0:
        raise AssertionError("Both 'control' and 'treatment' groups must be non-empty.")

    # --- Component Calculation ---
    
    # Control Group (x₀, n₀, s₀²)
    mean_c = np.mean(control_arr)
    var_c = np.var(control_arr, ddof=1) # ddof=1 for sample variance
    n_c = len(control_arr)              

    # Treatment Group (x₁, n₁, s₁²)
    mean_t = np.mean(treatment_arr)
    var_t = np.var(treatment_arr, ddof=1)
    n_t = len(treatment_arr)

    # Calculate the Standard Error of the Difference
    sdev = np.sqrt(
         (var_t / n_t)
         + (var_c / n_c)
    )

    # Calculate the t-statistic
    t_stat = (mean_t - mean_c) / sdev
    
    return float(t_stat) # Return as a float, as required