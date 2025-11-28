import numpy as np
# Se importa pandas aquí para que esté disponible globalmente para el Test 1.
# La verificación de tipos (PANDAS_SERIES) sigue siendo necesaria para el flujo.
try:
    import pandas as pd

    # Definir el tipo de Pandas Series aquí asegura que ARRAY_LIKE_TYPES
    # lo pueda utilizar correctamente.
    PANDAS_SERIES = pd.Series
except ImportError:
    # Si pandas no está disponible, el tipo es None para evitar errores.
    PANDAS_SERIES = type(None)


# Define a tuple with all accepted array-like types
ARRAY_LIKE_TYPES = (list, tuple, np.ndarray, PANDAS_SERIES)


def welch_t_stat(control, treatment):
    """
    Calculates the Welch's t-statistic for the difference in means between two 
    samples.
    
    The formula is: 
    $$t = \frac{\bar{x}_1 - \bar{x}_0}{\sqrt{s_1^2/n_1 + s_0^2/n_0}}$$
    """

    # 1. TYPE CHECK (Guardrail 1 - Required: TypeError)
    # Verifica si el objeto NO es uno de los tipos array-like esperados.
    if not isinstance(control, ARRAY_LIKE_TYPES):
        raise TypeError(
            "Expected 'control' to be an array-like object (e.g., list, "
            f"tuple, numpy.ndarray, pandas.Series), but got {type(control).__name__}"
        )

    if not isinstance(treatment, ARRAY_LIKE_TYPES):
        raise TypeError(
            "Expected 'treatment' to be an array-like object (e.g., list, "
            f"tuple, numpy.ndarray, pandas.Series), but got {type(treatment).__name__}"
        )

    # Convert to NumPy arrays for calculation
    control_arr = np.array(control)
    treatment_arr = np.array(treatment)

    # 2. SIZE CHECK (Guardrail 2 - AssertionError)
    if control_arr.size == 0 or treatment_arr.size == 0:
        raise AssertionError(
            "Both 'control' and 'treatment' groups must be non-empty."
        )

    # --- Component Calculation ---

    # Control Group (x0, n0, s0^2)
    mean_c = np.mean(control_arr)
    var_c = np.var(control_arr, ddof=1)  # ddof=1 for sample variance
    n_c = len(control_arr)

    # Treatment Group (x1, n1, s1^2)
    mean_t = np.mean(treatment_arr)
    var_t = np.var(treatment_arr, ddof=1)
    n_t = len(treatment_arr)

    # Calculate the Standard Error of the Difference
    sdev = np.sqrt((var_t / n_t) + (var_c / n_c))

    # Calculate the t-statistic
    t_stat = (mean_t - mean_c) / sdev

    return float(t_stat)  # Return as a float, as required