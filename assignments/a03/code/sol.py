

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

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    # Draw from generator
    return rng.normal(size=p) # Complete this line!


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
    z = X @ theta
    z = np.clip(z, -500, 500)
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
    probs = sigma(X, theta)
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
    y = np.asarray(y).ravel()
    p = sigma(X, theta)

    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())



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
    y = np.asarray(y).ravel()
    p = sigma(X, theta)          # (n,)
    n = y.shape[0]
    return (X.T @ (p - y)) / n   # (p,)

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
    # Init theta and evaluate the loss at that point
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    # Momentum state
    v = np.zeros_like(theta_0)
    beta = 0.9

    # Iterate from 1 to T
    theta_1 = theta_0  # safe default in case of immediate stop
    for i in range(max_iter):

        # Compute gradient and momentum update
        g = gradient(X=X, y=y, theta=theta_0)
        v = beta * v + (1.0 - beta) * g

        # Parameter update using momentum
        theta_1 = theta_0 - step_size * v

        # Evaluate new loss
        loss_1 = loss(X=X, y=y, theta=theta_1)

        # Early stopping rule
        if loss_0 - loss_1 < min_gain:
            break

        # Update for next iteration
        theta_0 = theta_1
        loss_0 = loss_1

    # Return final theta
    return theta_1


