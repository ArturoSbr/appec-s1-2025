"""Skeleton code to help get you started.

Do not modify this code because I will not grade it. Instead, make a copy of this file
and rename it `sol.py` and modify it there. Please note you don't need to add your
Student ID to it this time. The file should literally be called `sol.py`.

Anyway, this file contains all the functions you need to define. Together, they're meant
to work as the naive implementation of the gradient descent algorithm we saw in class.

See `../instructions.md` for more details!
"""

# Imports (you don't need anything else!)
import numpy as np


# Q1: Function that initializes theta
def init_theta(p, random_state):
    """Initialize theta using a standard normal distribution.

    This function draws `p` observations from a standard normal distribution.
    
    Parameters
    ----------
    p: int
        Number of parameters in the model. Must match the number of columns in the data.
    random_state: int
        Seed used to replicate draw (see numpy.random.default_rng).

    Returns
    -------
    Numpy ndarray of shape (p,).
    """
    # Validate input type
    if not isinstance(p,int):
            raise TypeError("p must be integer")

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    # Draw from generator
    return rng.normal(loc=0.0, scale=1.0, size=p)  # Complete this line!


# Q2. Vectorized function that calculates predicted probabilities
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
    # Validate input types
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(theta, np.ndarray):
        raise TypeError("theta must be a numpy.ndarray")

    # Performs matrix-vector multiplication
    z = X @ theta

    # Returns sigmoid function
    return 1 / (1 + np.exp(-z))


# Q3. Vectorized function that classifies observations based on their probability
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
    # Validate input types
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(theta, np.ndarray):
        raise TypeError("theta must be a numpy.ndarray")
    if not isinstance(threshold, float):
        raise TypeError("threshold must be a float")

    # Obtain predicted probabilities
    probs = sigma(X, theta)

    # Apply threshold validation
    return (probs > threshold).astype(int)

# Q4. Evaluate current loss given theta
def loss(X, y, theta):
    """Vectorized calculation of the negative log-likelihood given theta.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    y: numpy.ndarray
        Target of shape (n,) containing the observed labels of each observation.
    theta: numpy.ndarray
        The parameters of a logistic regression model. Its shape is (p,).

    Returns
    -------
    Float representing the loss value given the current estimates of theta.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(theta, np.ndarray):
        raise TypeError("theta must be a numpy.ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray")

    # Obtain predicted probabilities
    probs = sigma(X, theta)

    # Negative log-likelihood (cross-entropy)

    negll = -np.sum(y * np.log(probs) + (1-y) * np.log(1-probs))

    return negll

# Q5. Evaluate gradient
def gradient(X, y, theta):
    """Vectorized calculation of the gradient of the loss function at theta.
    
    Parameters
    ----------
    X: numpy.ndarray
        Feature matrix of shape (n, p) containing the observed characteristics of each
        observation. Its first column is always full of ones.
    y: numpy.ndarray
        Target of shape (n,) containing the observed labels of each observation.
    theta: numpy.ndarray
        The parameters of a logistic regression model. Its shape is (p,).

    Returns
    -------
    Numpy ndarray of shape (p,) representing the gradient of the loss function at theta.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(theta, np.ndarray):
        raise TypeError("theta must be a numpy.ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray")

    # Obtain predicted probabilities
    probs = sigma(X, theta)

    # Gradient
    grad = X.T @ (probs - y)

    return grad


# Q6. Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
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
        Target of shape (n,) containing the observed labels of each observation.
    random_state: int
        The seed used to initialize theta.
    max_iter: int
        The maximum number of updates to theta. Defaults to 10000.
    min_gain: float
        The minimum gain in the loss function to trigger another iteration. If the
        updated parameters, theta, fail to cause a gain larger than this threshold, the
        algorithm comes to an early stop. Defaults to 0.00001.
    step_size: Float
        This factor represents the fraction of the gradient used to update theta at each
        iteration (`theta_1 = theta_0 - step_size * nabla`). Defaults to 0.001.

    Returns
    -------
    Numpy ndarray of shape (p,) representing the final parameters estimated by the
    naive gradient descent algorithm.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy.ndarray")
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an int")
    if not isinstance(max_iter, int):
        raise TypeError("max_iter must be an int")
    if not isinstance(min_gain, float):
        raise TypeError("min_gain must be a float")
    if not isinstance(step_size, float):
        raise TypeError("step_size must be float")

    # Init theta and evaluate the loss at that point
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    # Iterate from 1 to T
    for i in range(max_iter):
        grad = gradient(X, y, theta_0)
        theta_1 = theta_0 - step_size * grad
        loss_1 = loss(X, y, theta_1)
        if loss_0 - loss_1 < min_gain:
            break
        
        # Update parameters and loss for next iteration
        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1
