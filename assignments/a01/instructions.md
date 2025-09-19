# Assignment 1
In this assignment you will add a Python script to a new branch. This Python script
should contain a custom function that calculates the _t_-statistic for the difference in
means between two samples. This statistic is commonly used to test for a statistically
significant treatment effect in a randomized trial.

On top of calculating an estimate, your function is expected to have a few guardrails to
help the user _use_ your function appropriately. Namely, your function should have a
couple of assertions and error messages.

## Instructions
1. Make sure you're standing on branch `develop` (`git checkout develop`).
2. Update `develop` (`git pull develop`)!
3. Create a new branch with the following format: `a01-<student ID here (6 digits)>`.
4. Create a new file in `assignments/a01/code` named `sol-<student ID here (6 digits)>`.
5. Declare a function named `welch_t_stat` in that file.
6. Add, commit and push your changes to trigger the autograder.

Your function must be called `welch_t_stat` and expect two arguments:
1. `control`
2. `treatment`

Both arguments are expected to be array-like objects, including lists, tuples, numpy
arrays and pandas seiers. If the user passes an incorrect object (a PyTorch tensor, for
example), the function is expected to raise a `TypeError`.

The _t_-statistic is calculated by subtracting the mean of the control group from that
of the test group and dividing by this new random variable's standard deviation. The null
hypothesis assumes that both means are equal and allows each group to have its own
variance. Hence:

$$
    t = \frac{
        \bar{x}_1 - \bar{x}_0
    }{
        \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}
    }
$$

Your function must calucate and return this statistic as a float.
