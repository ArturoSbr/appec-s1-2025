"""Skeleton code to 
help get 
you 
started.

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
    rng = np.random.default_rng(random_state)
    return rng.normal(size=p)


# Q4. Vectorized function that calculates predicted probabilities
def sigma(X, theta):
    z = X @ theta
    return 1 / (1 + np.exp(-z))


# Q5. Vectorized function that classifies observations
def classify(X, theta, threshold=0.5):
    probs = sigma(X, theta)
    return (probs > threshold).astype(int)


# Q6. Evaluate current loss given theta
def loss(X, y, theta):
    probs = sigma(X, theta)

    # Avoid log(0)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)

    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))


# Q7. Evaluate gradient
def gradient(X, y, theta):
    probs = sigma(X, theta)
    n = X.shape[0]
    return (X.T @ (probs - y)) / n


# Q8. Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
):
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X, y, theta_0)

    for _ in range(max_iter):
        grad = gradient(X, y, theta_0)
        theta_1 = theta_0 - step_size * grad
        loss_1 = loss(X, y, theta_1)

        if loss_0 - loss_1 < min_gain:
            break

        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1

