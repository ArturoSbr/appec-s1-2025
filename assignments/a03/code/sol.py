import numpy as np

# Q3: Function that initializes theta
def init_theta(p, random_state):
    """Initialize theta using a standard normal distribution."""
    # Initialize random number generator
    rng = np.random.default_rng(random_state)
    
    # Draw from generator (vectorized for p parameters)
    return rng.normal(size=p)


# Q4. Vectorized function that calculates predicted probabilities
def sigma(X, theta):
    """Vectorized calculation of predicted probabilities."""
    # Calculate z = X * theta
    z = X @ theta 
    # Apply sigmoid function: 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))


# Q5. Vectorized function that classifies observations
def classify(X, theta, threshold=0.5):
    """Classifies observations based on their predicted probabilities.
    
    This function calls `sigma` to calculate probabilities and classifies
    each observation using `threshold`.
    """
    # Get probabilities
    probs = sigma(X, theta)
    # Classify: 1 if prob > threshold, else 0
    # (astype(int) converts boolean to 0/1)
    return (probs > threshold).astype(int)


# Q6. Evaluate current loss given theta
def loss(X, y, theta):
    """Vectorized calculation of the negative log-likelihood given theta."""
    # Get predictions
    h = sigma(X, theta)
    
    # Small epsilon to avoid log(0) (gives -inf)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    # Negative Log-Likelihood Formula (Sum):
    # - sum( y*log(h) + (1-y)*log(1-h) )
    J = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    return J


# Q7. Evaluate gradient
def gradient(X, y, theta):
    """Vectorized calculation of the gradient of the loss function."""
    # Get predictions
    h = sigma(X, theta)
    # Gradient formula: X.T * (Predictions - Actual)
    # Note: If standard Log-Likelihood (sum), we do NOT divide by n.
    grad = X.T @ (h - y)
    return grad


# Q8. Main naive gradient descent function
def gradient_descent(
    X, y, random_state, max_iter=10000, min_gain=0.00001, step_size=0.001
):
    """Naive gradient descent algorithm for logistic regression."""
    # Init theta and evaluate the loss at that point
    p = X.shape[1]
    theta_0 = init_theta(p=p, random_state=random_state)
    loss_0 = loss(X=X, y=y, theta=theta_0)

    # Iterate from 1 to T
    for i in range(max_iter):

        # 1. Calculate gradient at current position (theta_0)
        grad = gradient(X, y, theta_0)

        # 2. Update theta (Gradient Descent Step)
        # theta_new = theta_old - learning_rate * gradient
        theta_1 = theta_0 - step_size * grad

        # 3. Calculate new loss with new theta
        loss_1 = loss(X, y, theta_1)

        # 4. Check for convergence (Stop if gain is too small)
        # We want loss to decrease, so loss_0 should be > loss_1.
        if loss_0 - loss_1 < min_gain:
            # Update final values before breaking
            theta_0 = theta_1
            break
        
        # 5. Update parameters and loss for next iteration
        theta_0 = theta_1
        loss_0 = loss_1

    # Return final theta
    return theta_0