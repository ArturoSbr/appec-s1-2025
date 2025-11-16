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
    Numpy ndarray of shape (p,).
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

    This function calls `sigma(X, theta)` to calculate the array of predicted
    probabilities and then classifies each observation using `threshold`. If a given
    probability is greater than `threshold`, it labels it as `1` and `0` otherwise.
    
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

    Returns
    -------
    Numpy ndarray of shape (n,) representing the predicted labels for each observation.
    """
    pass


# Q6. Evaluate current loss given theta
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

    Returns
    -------
    Float representing the loss value given the current estimates of theta.
    """
    pass


# Q7. Evaluate gradient
def gradient(X, y, theta):
    """Vectorized calculation of the gradient of the loss function at theta.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    y: numpy.ndarray
        Target of shape (n,) cointaining the observed labels of each observation.
    theta:
        The parameters of a logistic regression model. Its shape is (p,).

    Returns
    -------
    Numpy ndarray of shape (p,) representing the gradient of the loss function at theta.
    """
    pass


# Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=100000, min_gain=1e-6, step_size=0.001
):
    """Naive gradient descent algorithm for logistic regression.
    
    This function is an implementation of the original gradient descent algorithm for
    logistic regression. It minimizes the negative loss-likelihood cross entropy loss
    by iteratively updating theta using the loss's gradient.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    y: numpy.ndarray
        Target of shape (n,) cointaining the observed labels of each observation.
    random_state: int
        The seed used to initialize theta.
    max_iter: int
        The maximum number of updates to theta. Defaults to 100000.
    min_gain: float
        The minimum gain in the loss function to trigger another iteration. If the
        updated parameters, theta, fail to cause a gain larger than this threshold, the
        algorithm comes to an early stop. Defaults to 1e-6
    step_size: Float
        This factor represents the fraction of the gradient used to update theta at each
        iteration (`theta_1 = theta_0 - step_size * nabla`).

    Returns
    -------
    Numpy ndarray of shape (p,) representing the final parameters estimated by the
    naive gradient descent algorithm.
    """
    # Init theta randomly
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)

    # Iterate from 1 to T
    for i in range(max_iter):

        # Use previous function to calculate necessary inputs
        loss_0 = True
        nabla = True
        theta_1 = theta_0 - step_size * nabla
        loss_1 = True
        gain = abs(loss_1 - loss_0)  # Compare with min_gain to check for early stop
    
    # Return final theta (in case of no early stop)
    return theta_1
