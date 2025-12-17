import numpy as np


# Q3: Initialize theta
def init_theta(p, random_state):
    rng = np.random.default_rng(random_state)
    return rng.normal(size=p)


# Q4: Sigmoid / predicted probabilities
def sigma(X, theta):
    z = X @ theta
    return 1 / (1 + np.exp(-z))


# Q5: Classification rule
def classify(X, theta, threshold=0.5):
    probs = sigma(X, theta)
    return (probs > threshold).astype(int)


# Q6: Negative log-likelihood (cross-entropy loss)
def loss(X, y, theta):
    probs = sigma(X, theta)

    # Numerical stability
    eps = 1e-9
    probs = np.clip(probs, eps, 1 - eps)

    return -np.mean(
        y * np.log(probs) + (1 - y) * np.log(1 - probs)
    )


# Q7: Gradient of the loss
def gradient(X, y, theta):
    probs = sigma(X, theta)
    return X.T @ (probs - y) / X.shape[0]


# Q8: Gradient Descent
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
