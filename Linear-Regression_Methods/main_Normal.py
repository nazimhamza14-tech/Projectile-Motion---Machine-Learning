# Imports
import numpy as np
from Linear_Regression_Normal import LinearRegressionNormal

# CONSTANTS, this is a quadratic example in the form ax^2 + bx + c
START_X = -50
END_X = 50
NO_OF_PLOTS = 100
A_TERM = 3
B_TERM = -5
C_TERM = 2
DEGREE = 2

# Initialization
x = np.linspace(START_X, END_X, NO_OF_PLOTS)
y = np.array(A_TERM*x**DEGREE + B_TERM*x + C_TERM)


# Calling and using the model
LRNormal = LinearRegressionNormal(x, y)
LRNormal.restructure(DEGREE)
LRNormal.calc_weights()
LRNormal.results()
