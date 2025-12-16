"""Skeleton code to help get you started.
# Student ID: 189610

Do not modify this code because I will not grade it. Instead, make a copy of this file
and rename it `sol.py` and modify it there. Please note you don't need to add your
Student ID to it this time. The file should literally be called `sol.py`.

Anyway, this file contains all the functions you need to define. Together, they're meant
to work as the naive implementation of the gradient descent algorithm we saw in class.

See `../instructions.md` for more details!
"""


# Imports (you don't need anything else!)
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
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)

    return out


# Q5. Vectorized function that classifies observations based on their probability
def classify(X, theta, threshold=0.5):
    """Classify observations using a probability threshold."""
    p_hat = sigma(X, theta)
    return (p_hat > threshold).astype(int)


# Q6. Evaluate current loss given theta
def loss(X, y, theta):
    """Negative log-likelihood (cross-entropy loss)."""
    p_hat = sigma(X, theta)
    eps = 1e-15
    p_hat = np.clip(p_hat, eps, 1.0 - eps)
    return float(-np.sum(y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat)))


# Q7. Evaluate gradient
def gradient(X, y, theta):
    """Gradient of the loss function."""
    p_hat = sigma(X, theta)
    return X.T @ (p_hat - y)


# Q8. Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
):
    """Naive gradient descent algorithm for logistic regression."""
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    for _ in range(max_iter):
        theta_1 = theta_0 - step_size * gradient(X=X, y=y, theta=theta_0)
        loss_1 = loss(X=X, y=y, theta=theta_1)

        if loss_0 - loss_1 < min_gain:
            break

        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1
