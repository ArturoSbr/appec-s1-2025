# Assignment 3

This is your final assignment and is due on December 19th at 23:59:59 CST.

Your job is to implement your own version of the gradient descent algorithm to find the
best parameters of any logistic regression model.

It is extremely important that you vectorize your functions! Avoid using `for` loops at
all costs! The only `for` loop you should use is the one in the pseudo-code described
below.

## Instructions

1. Set the repo as your working directory: `cd path/to/repo/here`.
2. Checkout to branch `develop`: `git checkout develop`.
3. Update `develop`: `git pull origin develop`.
4. Create your own branch: `git checkout -b assignment/a03-<student ID>`. For
example: `git checkout -b assignment/a03-130524`.
5. Make a copy of `assignments/a03/code/hints.py` and name it `sol.py`. You can
do this with `cp assignments/a03/code/hints.py assignments/a03/code/sol.py`
from your terminal. Alternatively, you can right click on the `hints.py` file,
create a copy in the same folder and rename it to `sol.py`. Up to you.
6. Modify `sol.py` accordingly.

Like in any other exam, you won't know your final grade. However, you can
**estimate** how well you'll do by running a diagnostics file. When you're done
modifying your `sol.py` file, open a new terminal and:
```
$ cd <path/to/repo/here>  # Set repo as working directory
$ conda activate appec  # Activate virtual environment we created in September
$ python assignments/a03/code/diagnostic.py  # Execute this file
```

In order to test your algorithm, I will pass it the following arguments:

- $\mathbb{X} \in \mathbb{R}^{n \times p}$ ($\mathbb{X}$ **always** includes a column
full of ones for the intercept, so you don't need to add one yourself)
- $y \in \mathbb{R}^{n}$
- `random_state=42`
- `max_iter=10000`
- `min_gain=0.00001`
- `step_size=0.001`

Your algorithm should return a numpy ndarray of size $p$ representing the final
parameters $\theta^* \in \text{arg min } J(\theta)$.

### How to modify `sol.py`

1. Write a function `init_theta(p, random_state)` which draws `p` observations from a
standard normal distribution ($\mu = 0$ and $\sigma = 1$).

2. Write a function `sigma(X, theta)` that applies the sigmoid function to all the
observation (rows) of the dataset (`X`) using parameters `theta`. This function should
return a numpy ndarray of shape `(n,)`, where each element in the array represents the
predicted probability of each observation given `theta`.

3. Write a function `classify(X, theta, threshold)` that calls `sigma(X, theta)` to
calculate an array of predicted probabilities and then labes them using `threshold`. If
a given probability is greater than `threshold`, label it `1` and `0` otherwise.

4. Write a function `loss(X, y, theta)` that evaluates the negative log-likelihood
using cross-entropy. This function must return a real number representing the negative
log-likelihood given the current parameters `theta`.

5. Write a function `gradient(X, y, theta)` that evaluates the gradient at `X`, `y` and
`theta`. It should return a numpy ndarray of shape `(p,)`, and will be used to update
your parameters at each step of the optimization process.

6. Write a function
`gradient_descent(X, y, random_state, max_iter, min_gain, step_size)` that implements
the naive gradient descent algorithm seen in class. This function should return a numpy
ndarray of shape `(p,)` representing the final parameters found by your algorithm.

## Getting started

Take a look at `./code/hints.py` to get started. This file contains a basic skeleton.
All you need to do is create a copy (steps 1 and 2 above) and modify that file.
