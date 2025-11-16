"""Skeleton code to help get you started.

Do not modify this code because I will not grade it. Instead, make a copy of this file
and rename it `sol-<student ID>.py` and modify it there.

Anyway, this file contains all the functions you need to define. Together, they're meant
to work as the naive implementation of the gradient descent algorithm we saw in class.

See `../instructions.md` for more details!
"""

# Imports (you don't need any other libraries!)
import numpy as np
from scipy.stats import norm 


# Q3: Function that initializes theta
def init_theta(p, random_state):
    """Initialize theta using a multivariate normal distribution.
    
    Parameters
    ----------
    p: int
        Number of parameters in the model. Must match the number of columns in the data.
    random_state: int
        Seed used for replicability purposes.

    Returns
    -------
    ndarray of shape (p,).
    """
    pass


# Q4. Vectorized function that calculates predicted probabilities
def sigma(X, theta):
    """Vectorized calculation of predicted probabilities.

    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    theta: numpy.ndarray
        The parameters of a logistic regression model. Its shape is (p,).

    Returns
    -------
    Numpy ndarray of shape (n,) representing the predicted probabilities for each
    observation.
    """
    pass


# Q5. Vectorized function that classifies observations based on their probability
def classify(X, theta, threshold=0.5):
    """Classifies all observations in the data based on their predicted probabilities.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    theta: numpy.ndarray
        The parameters of a logistic regression model. Its shape is (p,).
    threshold: float
        A float between 0.0 and 1.0. Observations whose predicted probabilities are
        strictly greater than this threshold are classified as 1 and 0 otherwise.
        Defaults to 0.5.
    """
    pass


# Evaluate loss
def loss(X, y, theta):
    """Vectorized calculation of the negative log-likelihood given theta.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    y: numpy.ndarray
        Target of shape (n,) cointaining the observed labels of each observation.
    theta:
        The parameters of a logistic regression model. Its shape is (p,).
    """
    pass


# Evaluate gradient
def gradient(X, y, theta):
    pass


# Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=100000, min_gain=1e-6, step_size=0.001
):
    pass