# imports
import numpy as np
from Linear_RegressionGD import LinearRegressionGD
from Linear_Regression_Normal import LinearRegressionNormal


# CONSTANTS
START_X = -10
END_X = 10
NO_OF_PLOTS = 20
GRADIENT = 3
INTERCEPT = 4
DEGREE = 2
LEARNING_RATE = 0.0001
EPOCHS = 100000

# Use if attempting polynomial regression by wrapping the initialisation of x and y with scale(np. ...)
def scale(arr):
    """
    A function that scales the values to between 0 and 1 to avoid overflow errors when attempting polynomial regression

    Args:
        arr (1D array): An array of the values to be scaled

    Returns:
        1D array: (arr) of the scaled data
    """

    # Equation used to scale the data
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr

# Initialization
x = np.linspace(START_X, END_X, NO_OF_PLOTS)
y = np.array(x**2 + 4)

# Calling and using the gradient descent model
LR = LinearRegressionGD(x, y, LEARNING_RATE, EPOCHS)
LR.restructure(DEGREE)
LR.train()
LR.results()
