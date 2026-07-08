# Imports
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionNormal:
    """
    A linear regression model that uses the closed form solution (normal equation) to find the optimal weights for a
    given line in one calculation.
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def restructure(self, degree):
        """
        A function that restructures the input data that will allow for the vectorized calculation of predicted values
        and partial derivatives

        Args:
            degree (int): The degree of polynomial the data is governed by
        """

        features = self.features.T
        vectors = []
        # Creating the data matrix where each column represents the data to a certain power
        for power in range(degree + 1):
            degree_power = features**power
            vectors.append(degree_power)

        self.data = np.column_stack(vectors)
        self.labels = self.labels.reshape(-1, 1)

    def calc_weights(self):
        """
        A function that uses the normal equation to calculate the optimal weights
        """
        self.optimal_weights = np.linalg.inv((self.data.T @ self.data)) @ self.data.T @ self.labels
        self.optimal_weights = self.optimal_weights.reshape(-1, 1)

    def results(self):
        """
        A function to evaluate how the model performed by showing the initial weights, final weights and a graph
        comparing the models predictions with the actual values.
        """

        predictions = self.data @ self.optimal_weights

        # A column vector of the coefficients, going from top to bottom the power of x increases
        print(f"Predicted weights:\n {self.optimal_weights}")

        plt.plot(self.features, self.labels, color="blue", label="Actual graph")
        plt.plot(self.features, predictions, color="red", label="Predicted graph")
        plt.legend()
        plt.title("Model evaluation")
        plt.show()
