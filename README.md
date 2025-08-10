# Applied Econometrics I
Welcome to **Applied Econometrics I**! As the title suggests, this is the first course
of a two-semester journey designed to teach you the theory and skills needed to conduct
robust empirical analysis in economics and related fields.

While this is an "applied" course, our first semester will mainly focus on the
**theoretical principles** of the models we will use next semester.

## Our guiding principle
> **In order to be a good applied scientist, you must understand the theory behind the
> tools you use.**

## Syllabus
1. Tech skills
  - Bash
  - Git
  - GitHub
  - Python
2. Linear Regression
  - Assumptions
  - Interpretation
3. Maximum Likelihood Estimation
  - Gradient Descent
4. Generalized Linear Models
  - Probit
  - Logit

## Assignments
Assignments will be autograded by a set of GitHub Actions that are triggered when
you submit your code. More details coming soon!

## Structure of this repo
```
.
├── .flake8          # Linting configuration
├── .github          # GitHub builds (ignore this)
├── .gitignore       # Files ignored by git (ignore this)
├── assignments      # Homework assignments
├── environment.yml  # Virtual environment
├── lecture-notes    # Lecture notes
└── README.md        # Summary of this repo
```

## Virtual Environment
A virtual environment is a Python configuration designed to make your code replicable
across computers. The idea is that my code should run on your computer the same way it
does on mine.

This repo's configuration is very similar to
[Google Colab](https://colab.research.google.com), so if you forget your computer or you
simply cannot replicate our environment, you can always go use Colab as a last resort.

### Setting up your virtual environment
If you're doing this for the first time, follow all these steps (or skip the ones you've
already done before, such as installing Miniconda). If you've already done this before
and all you want to do is activate the environment, then just run step 7.

1. Go to the [Anaconda Downloads page](https://www.anaconda.com/download/success).
2. Download the corresponding Miniconda Graphical Installer for your computer.
3. Install the program you just downloaded.
4. Clone this repository.
5. Navigate into the cloned repository.
6. Run `conda env create -f environment.yml` (this uses the file `environment.yml` and
installs everything that's in it).
7. Activate the environment with `conda activate econ`.
