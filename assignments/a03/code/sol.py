
import numpy as np

# Q3: Function that initializes theta
def init_theta(p, random_state):
    rng = np.random.default_rng(random_state)
    return rng.normal(size=p)

# Q4. Vectorized function that calculates predicted probabilities
def sigma(X, theta):
    z = X @ theta
    return 1 / (1 + np.exp(-z))

# Q5. Vectorized function that classifies observations based on their probability
def classify(X, theta, threshold=0.5):
    predictions = sigma(X, theta)
    return (predictions > threshold).astype(int)

# Q6. Evaluate current loss given theta
def loss(X, y, theta):
    n = len(y)
    h = sigma(X, theta)
    epsilon = 1e-15
    cost = -(1/n) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# Q7. Evaluate gradient
def gradient(X, y, theta):
    n = len(y)
    h = sigma(X, theta)
    return (1/n) * (X.T @ (h - y))

# Q8. Main naive gradient descent function
def gradient_descent(X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001):
    theta_0 = init_theta(p=X.shape[1], random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    for i in range(max_iter):
        grad = gradient(X, y, theta_0)
        theta_1 = theta_0 - step_size * grad
        loss_1 = loss(X, y, theta_1)

        if loss_0 - loss_1 < min_gain:
            return theta_1
        
        theta_0 = theta_1
        loss_0 = loss_1

    return theta_1