import unittest
import pandas as pd
import numpy as np
import glob
import importlib.util
import sys
import os
import subprocess

class TestAssignment2(unittest.TestCase):
    student_module = None
    student_file = None
    
    # Scoring tracking
    score = 0
    total_tests = 12

    @classmethod
    def setUpClass(cls):
        """
        Locates and imports the student's solution file.
        """
        # 1. Find the student's file pattern sol-*.py
        student_files = glob.glob("sol-*.py")
        
        if len(student_files) == 0:
            raise FileNotFoundError("No file matching 'sol-*.py' found in the current directory.")
        elif len(student_files) > 1:
            print(f"Warning: Multiple solution files found: {student_files}. Testing {student_files[0]}.")
        
        cls.student_file = student_files[0]
        print(f"Grader: Loading {cls.student_file}...")

        # 2. Dynamically import the student's script as a module
        spec = importlib.util.spec_from_file_location("student_sol", cls.student_file)
        cls.student_module = importlib.util.module_from_spec(spec)
        
        # Execute the module to populate variables
        try:
            spec.loader.exec_module(cls.student_module)
        except Exception as e:
            print(f"CRITICAL: Student script crashed during execution: {e}")
            print("Grader will attempt to test variables created before the crash.")

    @classmethod
    def tearDownClass(cls):
        """
        Prints the final grade after all tests have run.
        """
        print("\n" + "="*30)
        print(f"Final grade: {cls.score}/{cls.total_tests}")
        print("="*30 + "\n")

    def get_variable(self, var_name):
        """Helper to safely get a variable from the student's code."""
        if not hasattr(self.student_module, var_name):
            self.fail(f"Variable '{var_name}' not found in your submission.")
        return getattr(self.student_module, var_name)

    # ------------------------------------------------------------------
    # Q1 - Q3: Data Preparation
    # ------------------------------------------------------------------

    def test_q01_happy_column(self):
        """Q1: Checks 'happy' column creation logic."""
        df = self.get_variable('table_reviews')
        
        if 'happy' not in df.columns:
            self.fail("Column 'happy' not found in table_reviews.")
            
        # FIX: We cannot check the count (99224) because Q2 filters rows later.
        # Instead, we verify the LOGIC matches the requirement on the remaining rows.
        # Logic: happy == 1 if review_score >= 4, else 0.
        
        expected_happy = df['review_score'].ge(4).astype(int)
        
        # Check if student's column matches our recalculation
        matches = (df['happy'] == expected_happy).all()
        self.assertTrue(matches, "Q1: Logic for 'happy' column is incorrect (values do not match review_score >= 4).")
        
        # If we get here, pass
        TestAssignment2.score += 1

    def test_q02_reviews_filtering(self):
        """Q2: Checks filtering and sorting of table_reviews."""
        df = self.get_variable('table_reviews')
        
        self.assertEqual(df.shape[0], 98673, "Q2: table_reviews row count is incorrect.")
        self.assertEqual(df['order_id'].duplicated().sum(), 0, "Q2: Duplicate order_ids found in table_reviews.")
        
        # Edge Case Timestamp
        target_id = 'df56136b8031ecd28e200bb18e6ddb2e'
        try:
            actual_ts = df.loc[df['order_id'] == target_id, 'review_answer_timestamp'].iloc[0]
            expected_ts = pd.Timestamp("2017-02-14 13:58:48")
            self.assertEqual(actual_ts, expected_ts, f"Q2: Incorrect timestamp for order {target_id}")
        except IndexError:
            self.fail(f"Q2: Test order_id '{target_id}' not found in dataframe.")
            
        TestAssignment2.score += 1

    def test_q03_aggregation(self):
        """Q3: Checks aggregation logic in agg_items."""
        df = self.get_variable('agg_items')
        
        expected_cols = ['n_items', 'avg_price', 'avg_shipping']
        for col in expected_cols:
            if col not in df.columns:
                self.fail(f"Q3: Column '{col}' missing from agg_items.")

        self.assertAlmostEqual(df.shape[0], 98666, delta=5, msg="Q3: Row count incorrect.")
        self.assertAlmostEqual(df['avg_price'].mean(), 125.919, places=2, msg="Q3: avg_price mean incorrect.")
        
        TestAssignment2.score += 1

    # ------------------------------------------------------------------
    # Q4 - Q6: Merging and Feature Engineering (Part 1)
    # ------------------------------------------------------------------

    def test_q04_initial_merge_rows(self):
        """Q4: Checks row integrity of the main df."""
        df = self.get_variable('df')
        # Row count must be consistent throughout
        self.assertEqual(df.shape[0], 97917, "Q4: df row count is incorrect.")
        self.assertEqual(df['order_id'].duplicated().sum(), 0, "Q4: Duplicate order_ids found in df.")
        
        TestAssignment2.score += 1

    def test_q05_days_delay_calc(self):
        """Q5: Checks 'days_delay' calculation in table_orders."""
        df = self.get_variable('table_orders')
        
        if 'days_delay' not in df.columns:
            self.fail("Q5: 'days_delay' column not found in table_orders.")
            
        self.assertAlmostEqual(df['days_delay'].count(), 96476, delta=10, msg="Q5: 'days_delay' count is incorrect.")
        self.assertAlmostEqual(df['days_delay'].mean(), -11.877, places=2, msg="Q5: 'days_delay' mean is incorrect.")
        
        TestAssignment2.score += 1

    def test_q06_merge_delay_status(self):
        """Q6: Checks merge of days_delay and order_status."""
        df = self.get_variable('df')
        
        # Check specific Q6 columns exist
        if 'days_delay' not in df.columns or 'order_status' not in df.columns:
            self.fail("Q6: 'days_delay' or 'order_status' missing from df.")
            
        # Check NaNs as requested (2087 indicates pending orders)
        self.assertEqual(df['days_delay'].isna().sum(), 2087, "Q6: Incorrect number of NaNs in 'days_delay'.")
        
        TestAssignment2.score += 1

    # ------------------------------------------------------------------
    # Q7 - Q9: Feature Engineering (Part 2)
    # ------------------------------------------------------------------

    def test_q07_photo_aggregation(self):
        """Q7: Checks photo imputation and aggregation."""
        df = self.get_variable('agg_pics')
        
        if 'avg_pics' not in df.columns:
            self.fail("Q7: 'avg_pics' column missing from agg_pics.")

        self.assertAlmostEqual(df.shape[0], 98666, delta=5, msg="Q7: Row count incorrect.")
        self.assertAlmostEqual(df['avg_pics'].mean(), 2.232, places=3, msg="Q7: avg_pics mean incorrect.")
        
        TestAssignment2.score += 1

    def test_q08_merge_photos(self):
        """Q8: Checks merge of avg_pics."""
        df = self.get_variable('df')
        
        if 'avg_pics' not in df.columns:
            self.fail("Q8: 'avg_pics' missing from df.")
            
        # Check NaNs (Should be 0 due to imputation)
        self.assertEqual(df['avg_pics'].isna().sum(), 0, "Q8: NaNs found in 'avg_pics' column (Imputation failed?).")
        
        TestAssignment2.score += 1

    def test_q09_constant(self):
        """Q9: Checks creation of constant column."""
        df = self.get_variable('df')
        
        if 'const' not in df.columns:
            self.fail("Q9: 'const' column missing.")
        
        # Check value is 1
        self.assertTrue((df['const'] == 1).all(), "Q9: 'const' column contains values other than 1.")
        
        TestAssignment2.score += 1

    # ------------------------------------------------------------------
    # Q10 - Q11: Regression Models
    # ------------------------------------------------------------------

    def test_q10_logit_model_1(self):
        """Q10: Checks the first Logit model (m1_res)."""
        res = self.get_variable('m1_res')
        
        # Check observations
        self.assertEqual(res.nobs, 95824, "Q10: Number of observations (nobs) is incorrect.")
        
        # Check specific parameters (using approx equality for floats)
        params = res.params
        self.assertAlmostEqual(params['const'], 1.314973, places=4, msg="Q10: 'const' coefficient incorrect.")
        self.assertAlmostEqual(params['n_items'], -0.498888, places=4, msg="Q10: 'n_items' coefficient incorrect.")
        self.assertAlmostEqual(params['days_delay'], -0.063472, places=4, msg="Q10: 'days_delay' coefficient incorrect.")
        
        TestAssignment2.score += 1

    def test_q11_logit_model_2(self):
        """Q11: Checks the second Logit model (m2_res)."""
        res = self.get_variable('m2_res')
        
        # Check observations
        self.assertEqual(res.nobs, 86296, "Q11: Number of observations (nobs) is incorrect.")
        
        # Check specific parameters
        params = res.params
        self.assertAlmostEqual(params['const'], 0.769101, places=4, msg="Q11: 'const' coefficient incorrect.")
        self.assertAlmostEqual(params['avg_price'], 0.000106, places=5, msg="Q11: 'avg_price' coefficient incorrect.")
        self.assertAlmostEqual(params['days_delay'], -0.071962, places=4, msg="Q11: 'days_delay' coefficient incorrect.")
        
        TestAssignment2.score += 1

    # ------------------------------------------------------------------
    # Q12: Code Style
    # ------------------------------------------------------------------

    def test_q12_linting(self):
        """Q12: Checks code style using flake8 (max-line-length=88)."""
        if not self.student_file:
            self.fail("Q12: Could not determine student file to lint.")

        # Ensure flake8 is installed
        from shutil import which
        if which('flake8') is None:
            print("WARNING: 'flake8' not found. Q12 skipped (Install via 'pip install flake8').")
            return

        # Run flake8 as a subprocess
        cmd = ['flake8', '--max-line-length=88', self.student_file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Formatting the output for clarity
            errors = result.stdout.strip()
            msg = f"Q12: flake8 found style errors:\n{errors}"
            self.fail(msg)
            
        TestAssignment2.score += 1

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)
    