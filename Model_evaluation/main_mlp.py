import numpy as np
import matplotlib.pyplot as plt
from Machine_Learning_models.mlp import NeuralNetwork
from utils import normalise, denormalise
from config import dt, D, mass, N, learning_rate
from PM_Methods.vectorized_RK4 import projectile_trajectory

def run_network():
    # Initialising training data by running simulation
    rng = np.random.default_rng()
    velocities = rng.uniform(low=5, high=75, size=N)
    train_angles = rng.uniform(low=10, high=80, size=N)

    ranges, max_heights = projectile_trajectory(velocities, train_angles, D, mass, dt, N)[0], projectile_trajectory(velocities, train_angles, D, mass, dt, N)[1]

    # Data is normalised to improve the networks performance
    velocities = normalise(velocities)
    ranges, max_heights = normalise(ranges), normalise(max_heights)

    # Initialising test data
    test_velocities = rng.uniform(low=5, high=75, size=N)
    test_angles = rng.uniform(low=10, high=80, size=N)

    # Saving the initial data to be used in plotting and minmax denormalisation
    test_ranges, test_max_heights = projectile_trajectory(test_velocities, test_angles, D, mass, dt, N)
    test_range_data, test_height_data = test_ranges, test_max_heights
    test_velocities = normalise(test_velocities)

    # Setting up + training the network
    neural_network = NeuralNetwork(learning_rate)
    neural_network.initialisation()

    training_labels = np.column_stack((ranges, max_heights))

    sin_train_angles = np.sin(np.deg2rad(train_angles))
    cos_train_angles = np.cos(np.deg2rad(train_angles))

    # Holding each cost for plotting
    range_cost = []
    height_cost = []

    # Setting up for the input data
    training_inputs = np.column_stack((velocities, sin_train_angles, cos_train_angles))

    # Training loop
    for _ in range(N):
        # Adjusts weights N times
        range_error = 0
        height_error = 0
        for i in range(N):

            training_a0 = training_inputs[i].reshape(-1, 1)
            current_label = training_labels[i].reshape(-1, 1)

            z3 = neural_network.forward_pass(training_a0)[4]

            range_error += neural_network.cost(z3, current_label)[0]
            height_error += neural_network.cost(z3, current_label)[1]

            # Slightly adjusts the weights each pass using backpropagation
            neural_network.backwards_pass(learning_rate, current_label, z3, training_a0)

        # Saving the average error for range and max height
        range_error, height_error = range_error/N, height_error/N
        range_cost.append(range_error)
        height_cost.append(height_error)

    # Testing + evaluation of the network
    sin_test_angles = np.sin(np.deg2rad(test_angles))
    cos_test_angles = np.cos(np.deg2rad(test_angles))

    range_predictions = []
    max_height_predictions = []

    test_inputs = np.column_stack((test_velocities, sin_test_angles, cos_test_angles))

    # Testing loop
    for i in range(N):
        test_a0 = test_inputs[i].reshape(-1, 1)

        z3 = neural_network.forward_pass(test_a0)[4]

        # Saving models prediction of range and max heights for plotting
        range_predictions.append(z3[0][0])
        max_height_predictions.append(z3[1])

    range_predictions, max_height_predictions = np.array(range_predictions), np.array(max_height_predictions)
    range_predictions = denormalise(range_predictions, np.max(test_range_data), np.min(test_range_data))
    max_height_predictions = denormalise(max_height_predictions, np.max(test_height_data), np.min(test_height_data))

    # Plotting the loss curves, as well as comparing predictions with real values
    passes = np.arange(1, N + 1)

    plt.plot(passes, range_cost, label="Range Loss")
    plt.xlabel("No. of passes")
    plt.ylabel("Range cost")
    plt.title("Range error evaluation")
    plt.legend()
    plt.show()

    plt.plot(passes, height_cost, label="Height Loss", color="red")
    plt.xlabel("No. of passes")
    plt.ylabel("Maximum height cost")
    plt.title("Maximum height error evaluation")
    plt.legend()
    plt.show()

    plt.scatter(test_range_data, range_predictions)
    plt.xlabel("Actual ranges")
    plt.ylabel("Predicted ranges")
    plt.title("Numerical range evaluation")
    plt.show()

    plt.scatter(test_height_data, max_height_predictions, color="red")
    plt.xlabel("Actual maximum heights")
    plt.ylabel("Predicted maximum heights")
    plt.title("Numerical maximum height evaluation")
    plt.show()

if __name__ == "__main__":
    run_network()