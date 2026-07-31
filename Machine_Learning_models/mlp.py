import numpy as np

class NeuralNetwork:

    def __init__(self,learning_rate):

        self.learning_rate = learning_rate

    def initialisation(self, input_layer=3, layer_1=8, layer_2=4, output_layer=2):

        # Initializing random weights with normal distribution then adjusting the standard deviation (He initialisation)
        self.w1 = np.random.randn(layer_1, input_layer) * np.sqrt(2 / input_layer)
        self.w2 = np.random.rand(layer_2, layer_1) * np.sqrt(2 / layer_1)
        self.w3 = np.random.rand(output_layer, layer_2) * np.sqrt(2 / layer_2)

        self.b1, self.b2, self.b3 = np.zeros((layer_1, 1)), np.zeros((layer_2, 1)), np.zeros((output_layer, 1))
        self.b1, self.b2, self.b3 = self.b1.reshape(-1, 1), self.b2.reshape(-1, 1), self.b3.reshape(-1, 1)

    def ReLU(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def ReLU_deriv(self, x):

        return np.where(x < 0, 0, 1)


    def forward_pass(self, a0):
        z1 = self.w1 @ a0 + self.b1
        a1 = self.ReLU(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = self.ReLU(z2)
        z3 = self.w3 @ a2 + self.b3

        return z1, a1, z2, a2, z3

    def cost(self, z3, labels):
        return (1/len(labels)) * (labels - z3)**2


    def backwards_pass(self, learning_rate, labels, z3, a0):
        z1, a1, z2, a2, z3 = self.forward_pass(a0)
        n = len(labels)

        jw3 = (-2/n) * (labels - z3) @ a2.T
        jb3 = (-2/n) * (labels - z3)

        jw2 = (((-2 / n) * (labels - z3)).T @ self.w3 * self.ReLU_deriv(z2).T).T @ a1.T
        jb2 = ((-2 / n) * (labels - z3)).T @ self.w3 * self.ReLU_deriv(z2).T

        jw1 = ((((-2 / n) * (labels - z3)).T @ self.w3 * self.ReLU_deriv(z2).T) @ self.w2 * self.ReLU_deriv(z1).T).T @ a0.T
        jb1 = (((-2 / n) * (labels - z3)).T @ self.w3 * self.ReLU_deriv(z2).T) @ self.w2 * self.ReLU_deriv(z1).T

        self.w3, self.w2, self.w1 = self.w3 - (learning_rate * jw3), self.w2 - (learning_rate * jw2), self.w1 - (learning_rate * jw1)
        self.b3, self.b2, self.b1 = self.b3 - (learning_rate * jb3), self.b2 - (learning_rate * jb2.T), self.b1 - (learning_rate * jb1.T)
