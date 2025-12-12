"""Run a diagnostic test to see if your solution is good enough.

This script will import two functions from your solution file:
1. `classify`
2. `gradient_descent`

Like in a normal exam, you won't know what grade you're going to get. However,
if your algorithm is good, then the grade you get in this diagnostic test
will be VERY close to the final grade you'll get once I apply it to a testing
dataset.

To run this file, open a new terminal and:
```
$ cd <path/to/repo/here>  # Set repo as working directory
$ conda activate appec  # Activate virtual environment we created in September
$ python assignments/a03/code/diagnostic.py  # Execute this file
```
"""

# Import your functions
import os
import pandas as pd
from sklearn.metrics import f1_score
from sol import classify, gradient_descent

# Log message
MSG = """Your algorithm achieved an F1 score of {f1:.3f} on the training data.
The testing data is very similar to the training data, so your algorithm should
perform very similarly when applied to it. This means your final grade should
be close to {proxy:.1f}/10.
You can try as many times as you want. It's up to you to decide if that's good
enough or if you want to keep trying.

Hope you enjoyed the course!"""


# Function that estimates your grade
def estimate_final_grade(f1):
    scaled = (f1 + 0.1) * 10
    grade = min(10, max(6, scaled))
    print('\n', MSG.format(f1=f1, proxy=grade), '\n', sep='')


# Load training data and turn it to np.ndarrays
_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_dir, '..', 'data', 'train.csv'))
X = df.loc[:, :'x5'].values
y = df['y'].values

# Use your algorithm to get the best parameters
theta = gradient_descent(
    X=X, y=y, random_state=42,
    max_iter=10000, min_gain=0.00001, step_size=0.001
)

# Use your parameters to classify each observation
pred = classify(X=X, theta=theta, threshold=0.5)

# Calculate your F1 score on the training data
f1 = f1_score(y_true=y, y_pred=pred)

# Estimate your final grade
estimate_final_grade(f1=f1)
