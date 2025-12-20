"""Skeleton code to help get you started.

Do not modify this code because I will not grade it. Instead, make a copy of this file
and rename it `sol.py` (or whatever your instructions say) and modify it there.

Anyway, this file contains all the functions you need to define. Together, they're meant
to work as the naive implementation of the gradient descent algorithm we saw in class.

See `../instructions.md` for more details!
"""

import numpy as np


# Q3: Function that initializes theta
def init_theta(p, random_state):
    """Initialize theta using a standard normal distribution."""
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.0, scale=1.0, size=p)


# Q4. Vectorized function that calculates predicted probabilities
def sigma(X, theta):
    """Vectorized calculation of predicted probabilities."""
    z = X @ theta
    return 1.0 / (1.0 + np.exp(-z))


# Q5. Vectorized function that classifies observations based on their probability
def classify(X, theta, threshold=0.5):
    """Classifies all observations based on predicted probabilities."""
    probs = sigma(X, theta)
    return (probs > threshold).astype(int)


# Q6. Evaluate current loss given theta
def loss(X, y, theta):
    """Vectorized negative log-likelihood (cross-entropy) loss."""
    probs = sigma(X, theta)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(-np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


# Q7. Evaluate gradient
def gradient(X, y, theta):
    """Vectorized gradient of the negative log-likelihood."""
    probs = sigma(X, theta)
    return X.T @ (probs - y)


# Q8. Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
):
    """Naive gradient descent algorithm for logistic regression."""
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    for _ in range(max_iter):
        grad = gradient(X=X, y=y, theta=theta_0)
        theta_1 = theta_0 - step_size * grad
        loss_1 = loss(X=X, y=y, theta=theta_1)

        if loss_0 - loss_1 < min_gain:
            break

        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1
