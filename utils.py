import numpy as np

def normalise(arr):
    """
    A function that scales the values to between 0 and 1 to avoid overflow errors when attempting polynomial regression
    using min-max normalization.

    Args:
        arr (1D array): An array of the values to be scaled

    Returns:
        1D array: (arr) of the scaled data
    """

    # Equation used to scale the data
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr

def denormalise(arr, original_max, original_min):
    """
    A function that denormalises scaled data in order to view the models predictions of the numerical max height
    and range

    Args:
        arr (1D array): An array of the values to be denormalised

    Returns:
        1D array: (arr) of the denormalised data
    """

    # Re-arranged version of the min-max equation
    arr = (arr * (original_max - original_min) + original_min)
    return arr