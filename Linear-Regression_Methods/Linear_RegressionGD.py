# imports
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    A Linear Regression model built from scratch to determine the relationship between data - including linear
    or polynomial relationships.
    """
    def __init__(self, features, labels, learning_rate, epochs):
        self.features = features
        self.labels = labels
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Reshaping the labels into a column vector to be able to calculate the loss
        self.labels = self.labels.reshape((-1, 1))

    # Transforming the inputs to be used in the matrix maths depending on the degree polynomial used
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

        # Creating and transposing the data matrix to perform matrix multiplication with the loss
        self.data = np.column_stack(vectors)
        self.data_T = self.data.T

        # Saving the initial weights to be used at the end in the results() method
        self.initial_weights = np.ones((degree + 1, 1))
        self.weights = np.ones((degree + 1, 1))

    def grad(self):
        """
        A function that calculates the partial derivatives of the MSE with respect to each weight and groups them into
        one vector.

        Returns:
             gradMSE (array): The partial derivatives in one column vector
        """

        # Transposing and multiplying by the loss calculates the partial derivatives of each weight with respect to the MSE (mean squared error)
        self.predictions = self.data @ self.weights
        loss = self.labels - self.predictions

        # Combining the partial derivatives into one vector gradMSE
        gradMSE = -2 / len(self.data) * (self.data_T @ loss)
        return gradMSE


    def train(self):
        """
        A function that trains the model to find the optimal weights that produce the smallest loss, the learning rate
        determines how big the jumps are to find this minimum.
        """

        for _ in range(self.epochs):
            gradMSE = self.grad()
            self.weights -= self.learning_rate * gradMSE


    def results(self):
        """
        A function to evaluate how the model performed by showing the initial weights, final weights and a graph
        comparing the models predictions with the actual values.
        """

        print(f"Initial weights:\n {self.initial_weights}")
        print(f"Final weights:\n {self.weights}")

        plt.plot(self.features, self.labels, color="blue", label="Actual graph")
        plt.plot(self.features, self.predictions, color="red", label="Predicted graph")
        plt.legend()
        plt.title("Model evaluation")
        plt.show()





