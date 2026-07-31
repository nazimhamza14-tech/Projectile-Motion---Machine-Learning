# imports
import numpy as np
from Machine_Learning_models.linear_regressionGD import LinearRegressionGD


# CONSTANTS
START_X = -50
END_X = 50
NO_OF_PLOTS = 100
GRADIENT = 3
INTERCEPT = 4
DEGREE = 1
LEARNING_RATE = 0.0001
EPOCHS = 100000

# Initialization
x = np.linspace(START_X, END_X, NO_OF_PLOTS)
y = np.array(GRADIENT*x + INTERCEPT)

# Calling and using the model
LR = LinearRegressionGD(x, y, LEARNING_RATE, EPOCHS)
LR.restructure(DEGREE)
LR.train()
LR.results()
