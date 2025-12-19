import numpy as np


def init_theta(p, random_state):
    rng = np.random.default_rng(random_state)
    return rng.normal(size=p)


def sigma(X, theta):
    z = X @ theta
    return 1.0 / (1.0 + np.exp(-z))


def classify(X, theta, threshold=0.5):
    probs = sigma(X, theta)
    return (probs > threshold).astype(int)


def loss(X, y, theta):
    p = sigma(X, theta)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def gradient(X, y, theta):
    p = sigma(X, theta)
    return X.T @ (p - y)


def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
):
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    theta_1 = theta_0

    for _ in range(max_iter):
        theta_1 = theta_0 - step_size * gradient(X=X, y=y, theta=theta_0)
        loss_1 = loss(X=X, y=y, theta=theta_1)

        if loss_0 - loss_1 < min_gain:
            break

        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1