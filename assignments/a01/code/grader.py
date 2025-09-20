import os
import sys
import importlib.util
from pathlib import Path
import subprocess

# Add the student's directory to the Python path for importing
STUDENT_SCRIPT_DIR = '.'
sys.path.append(STUDENT_SCRIPT_DIR)

# Initialize the score
score = 0
total_tests = 0


# Helper function to run tests
def check_test(test_function, description):
    """A helper function to run a test and update the score."""
    global score, total_tests
    total_tests += 1
    print(f"Running Test {total_tests}: {description}... ", end="")
    try:
        test_function()
        print("Passed!")
        score += 1
    except Exception as e:
        print(f"Failed. Reason: {e}")
    finally:
        print("-" * 20)


# Find the student's solution file
student_file = None
for f in os.listdir(STUDENT_SCRIPT_DIR):
    if f.startswith('sol-') and f.endswith('.py'):
        student_file = f
        break

if not student_file:
    print("Error: Student solution file not found (e.g., 'sol-130524.py').")
    print("Final Score: 0/10")
    sys.exit(1)

# Dynamically import the student's module
try:
    spec = importlib.util.spec_from_file_location("student_solution", student_file)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    sys.modules['student_solution'] = student_module
    from student_solution import welch_t_stat
    print("Successfully imported student's function.")
except (ImportError, AttributeError) as e:
    print(f"Cannot import 'welch_t_stat' from '{student_file}'.")
    print(f"Reason: {e}")
    print("Final Score: 0/10")
    sys.exit(1)


# Test 0: Checks if pandas and numpy were imported
def test_imports():
    assert 'numpy' in sys.modules, "NumPy was not imported."
    assert 'pandas' in sys.modules, "Pandas was not imported."


# Test 1: Checks if the function can be imported
def test_function_import():
    assert callable(welch_t_stat), "'welch_t_stat' is not a callable function."


# Test 2: List input
def test_list_input():
    result = welch_t_stat(control=[1, 2, 3, 1, 2, 3], treatment=[7, 8, 9, 7, 8, 9])
    assert abs(result - 11.618950038622252) < 1e-6, f"Expected 11.6189..., got {result}"


# Test 3: Tuple input
def test_tuple_input():
    result = welch_t_stat(control=(1, 2, 3, 1, 2, 3), treatment=(7, 8, 9, 7, 8, 9))
    assert abs(result - 11.618950038622252) < 1e-6, f"Expected 11.6189..., got {result}"


# Test 4: Numpy array input
def test_ndarray_input():
    import numpy as np
    result = welch_t_stat(
        control=np.array([1, 2, 3, 1, 2, 3]),
        treatment=np.array([7, 8, 9, 7, 8, 9])
    )
    assert abs(result - 11.618950038622252) < 1e-6, f"Expected 11.6189..., got {result}"


# Test 5: Pandas Series input
def test_series_input():
    import pandas as pd
    result = welch_t_stat(
        control=pd.Series([1, 2, 3, 1, 2, 3]),
        treatment=pd.Series([7, 8, 9, 7, 8, 9])
    )
    assert abs(result - 11.618950038622252) < 1e-6, f"Expected 11.6189..., got {result}"


# Test 6: Check for TypeError with a string `control`
def test_type_error_control():
    try:
        welch_t_stat(control='asdf', treatment=[1, 2, 3])
        raise AssertionError("TypeError was not raised for string control input.")
    except TypeError:
        pass  # Success
    except Exception as e:
        raise AssertionError(f"Expected TypeError, but got {type(e).__name__} instead.")


# Test 7: Check for TypeError with a string `treatment`
def test_type_error_treatment():
    try:
        welch_t_stat(control=[1, 2, 3], treatment='asdf')
        raise AssertionError("TypeError was not raised for string treatment input.")
    except TypeError:
        pass  # Success
    except Exception as e:
        raise AssertionError(f"Expected TypeError, but got {type(e).__name__} instead.")


# Test 8: Check against CSV data
def test_csv_data():
    import pandas as pd
    # We need to make sure the data.csv file is in the right place
    data_path = Path(STUDENT_SCRIPT_DIR) / '../data/data.csv'
    if not data_path.is_file():
        raise FileNotFoundError(f"data.csv not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    result = welch_t_stat(
        control=df.loc[df['treatment'].eq(0), 'y'],
        treatment=df.loc[df['treatment'].eq(1), 'y']
    )
    assert abs(result - 4.981451030981089) < 1e-6, f"Expected 4.9814..., got {result}"


# Test 9: Lint the code using flake8
def test_flake8_linting():
    # Use subprocess to run flake8 on the student's file
    result = subprocess.run(
        ['flake8', student_file, '--max-line-length=88'],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        raise AssertionError(f"Linting failed. Errors:\n{result.stdout}")


# Run tests
check_test(test_imports, "Imports NumPy and Pandas")
check_test(test_function_import, "Imports function")
check_test(test_list_input, "Calculates t-stat with lists")
check_test(test_tuple_input, "Calculates t-stat with tuples")
check_test(test_ndarray_input, "Calculates t-stat with NumPy arrays")
check_test(test_series_input, "Calculates t-stat with Pandas Series")
check_test(test_type_error_control, "Raises TypeError for string control input")
check_test(test_type_error_treatment, "Raises TypeError for string treatment input")
check_test(test_csv_data, "Calculates t-stat with real-world CSV data")
check_test(test_flake8_linting, "Lints code with flake8 (max-line-length=88)")
print(f"\nFinal Score: {score}/{total_tests}")
