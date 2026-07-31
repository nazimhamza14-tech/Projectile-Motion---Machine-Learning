# Imports
import numpy as np
from Machine_Learning_models.linear_regression_normal import LinearRegressionNormal
from config import dt, mass, D, N
from PM_Methods.vectorized_RK4 import projectile_trajectory

# Function to initialise, run and test the model
def fit_and_evaluate(features, labels, degree):

    """
    A function to initialise, run and test the linear regression model

    :param features: A 1d array consisting of the features
    :param labels: The labels resulting from each feature
    :param degree: The degree polynomial the model should fit
    :return: None
    """

    model = LinearRegressionNormal(features, labels)
    model.restructure(degree)
    model.calc_weights()
    model.results()

if __name__ == "__main__":
    """
    Simulates projectile trajectory for a number of projectiles, obtains ranges and maximum heights, then engineers the features for linearity, 
    finally runs and tests the model
    """

    # Initialization
    rng = np.random.default_rng(seed=108)
    velocities = rng.uniform(low=5, high=75, size=N)
    angles = rng.uniform(low=10, high=80, size=N)
    degree = 1 # Feature engineering removes the non-linearity this the model is now just looking for a straight line with degree polynomial 1
    ranges, max_heights = projectile_trajectory(velocities, angles, D, mass, dt, N)

    # Feature engineering
    r_features = velocities**2 * np.sin(2 * np.deg2rad(angles)) # Values chosen based on a certain physics term involved in the equations for a projectile without drag
    h_features = velocities**2 * (np.sin(np.deg2rad(angles))**2) # Again chosen based on the equation but for height instead

    # Calling and using the model to predict ranges and max heights respectively
    fit_and_evaluate(r_features, ranges, degree)
    fit_and_evaluate(h_features, max_heights, degree)