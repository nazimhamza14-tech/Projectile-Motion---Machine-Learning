# Imports
import numpy as np
import matplotlib.pyplot as plt


# Constants
STARTING_X = 1
ENDING_X = 50
NO_OF_PLOTS = 100
GRADIENT = 3
INTERCEPT = 4
NO_OF_EPOCHS = 1000


# Creating the model
class MyLinearRegression:
    def __init__(self, x, y, epoch, LEARNING_CONSTANT = 0.0001):
        self.x = x
        self.y = y
        self.points = x.size
        self.LEARNING_CONSTANT = LEARNING_CONSTANT
        self.weight = 0
        self.bias = 1
        self.cost = None
        self.epoch = epoch

    # The model aims to obtain a weight and a bias such that there is no change in cost
    def cost_error(self):
        error = (self.y - (self.weight * self.x + self.bias)) **2 # Formula to calculate the cost
        total = np.sum(error)

        # Returns the cost as per the formula for the mean squared error (MSE) - the average squared distance between predicted values and actual values
        return total / self.points

    def train(self):
        # Trains the model by updating the weight and cost and prints the final weight and cost
        print(f"Initial weight (gradient): {self.weight}, initial bias {self.bias}")
        for i in range(self.epoch):
            self.update_weights()

        self.cost = self.cost_error()
        print(f"Final weight (gradient): {self.weight}, final bias: {self.bias}")
        print(f"Actual gradient: {GRADIENT}, actual intercept (bias) {INTERCEPT}")
        return self.weight, self.bias

    def update_weights(self):
        difference = None
        weight_deriv = 0
        bias_deriv = 0

        # This is the essence of gradient descent in linear regression. First, the difference between the actual y value and predicted y value is found
        difference = self.y - (self.weight * self.x + self.bias)

        # Differentiate the difference in order to find the gradient, this comes form the formula for the cost
        weight_deriv = np.sum(-2 * difference * self.x)

        # Differentiate again to find the y-intercept
        bias_deriv = np.sum(-2 * difference)

        # Adjust both the weight (gradient) and bias (intercept) ever so slightly, allowing the values to move closer and closer to the true value
        self.weight -= (weight_deriv / self.points) * self.LEARNING_CONSTANT # Finds average gradient across all data points, multiplies by the learning rate and subtracts
        self.bias -= (bias_deriv / self.points) * self.LEARNING_CONSTANT # Again finds the average intercept across all data points, multiples by the learning constant and subtracts


# Function used to check the effectiveness of the model given a number of epochs, the graph for the model as well as the predicted gradient and intercept is displayed
def check(x, y, epochs):

    model = MyLinearRegression(x, y, epochs)
    line_properties = model.train()

    predicted_gradient, predicted_intercept = line_properties[0], line_properties[1]

    predicted_x = np.linspace(STARTING_X, ENDING_X, NO_OF_PLOTS)
    predicted_y = (predicted_gradient * predicted_x) + predicted_intercept

    plt.plot(predicted_x, predicted_y, color = "red", label = "Predicted line")
    plt.plot(x, y, label = f"y = {GRADIENT}x + {INTERCEPT}")
    plt.legend()
    plt.title(f"After {NO_OF_EPOCHS} epochs")
    plt.show()


# Used to compare the predicted X and Y values to the actual values as stated below
x = np.linspace(STARTING_X, ENDING_X, NO_OF_PLOTS)
y = (GRADIENT * x) + INTERCEPT

check(x, y, NO_OF_EPOCHS)