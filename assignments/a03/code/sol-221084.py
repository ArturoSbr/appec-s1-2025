import numpy as np


# Q3
def init_theta(p, random_state):
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.0, scale=1.0, size=p)


# Q4
def sigma(X, theta):
    z = X @ theta
    return 1 / (1 + np.exp(-z))


# Q5
def classify(X, theta, threshold=0.5):
    probs = sigma(X, theta)
    return (probs > threshold).astype(int)


# Q6
def loss(X, y, theta):
    p = sigma(X, theta)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


# Q7
def gradient(X, y, theta):
    p = sigma(X, theta)
    return X.T @ (p - y)


# Q8
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=1e-5, step_size=0.001
):
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    for _ in range(max_iter):
        grad = gradient(X, y, theta_0)
        theta_1 = theta_0 - step_size * grad
        loss_1 = loss(X, y, theta_1)

        if loss_0 - loss_1 < min_gain:
            break

        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1
