# Assignment 3

This is your final assignment and is due on December 19th at 23:59:59 CST.

Your job is to implement your own version of the gradient descent algorithm to find the
best parameters of any logistic regression model.

It is extrememly important that you vectorize your functions! Avoid using `for` loops at
all costs! The only `for` loop you should use is the one in the pseudo-code described
below.

## Instructions

1. Create your own branch `assignment/a03-<student ID>`. Make sure your branch branches
out from the most updated version of `develop` (`git checkout develop`,
`git pull develop`, and `git checkout -b assignment/a03-<student ID>`).
2. Create a Python file in the `/code/` directory of this assignment and name it
`sol-<student ID>.py. This is where you'll write your functions.

I order to test your algorithm, I will pass it the following arguments:

- $\mathbb{X} \in \mathbb{R}^{n \times p}$
- $y \in \mathbb{R}^{n \times 1}$
- \textit{max\_iter} $\in \mathbb{N}$
- \textit{min\_gain} $\in \mathbb{R}^+$
- \textit{step\_size} $\in \mathbb{R}^+$

Your algorithm should return a numpy ndarray of size $p \times 1$ representing the final
parameters $\theta^* \in \text{arg min} J(\theta)$.

3. Write a function `init_theta(p, random_state)` which draws a single variate from a
multivariate normal distribution with parameters $\mu = 0$ and $\sigma = 1$. Feel free
to use scipy's `normal` module. This function must return a numpy ndarray of shape
`(p,)`. This array will be used as the starting parameters of your model.

4. Write a function `sigma(X, theta)` that applies the sigmoid function to all the
observation (rows) of the dataset (`X`) using parameters `theta`. This function should
return a numpy ndarray of shape `(n, )`, where each element in the array represents the
predicted probability of each observation given `theta`.

5. Write a function `loss(X, y, theta)` that evaluates the negative log-likelihood
using cross-entropy. This function must return a real number representing the negative
log-likelihood given the current parameters `theta`.

7. Write a function `gradient(X, y, theta)` that evaluates the gradient at `X`, `y` and
`theta`. It should return a numpy ndarray of shape `(p,)`, and will be used to update
your parameters at each step of the optimization process.

8. Write a function
`gradient_descent(X, y, random_state, max_iter, min_gain, step_size)` that implements
the naive gradient descent algorithm seen in class. This function should return a numpy
ndarray of shape `(p,)` representing the final parameters found by your algorithm.
