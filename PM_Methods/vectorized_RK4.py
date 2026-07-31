# Imports
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 9.81
AIR_DENSITY = 1.225


# Function to calculate drag parameter
def drag_parameter(drag_coefficient, area):
    """
    Calculating the drag parameter used in this method which combines the drag coefficient, air density and area.

    Args:
        drag_coefficient (float): The drag coefficient of the projectile, dependent on the shape of the projectile
        area (float): Area of the projectile in m^2
        air_density (float): Air density at sea level in kgm^-3

    Returns:
        The value (float) of the drag parameter
    """

    D = 0.5 * drag_coefficient * area * AIR_DENSITY
    return D


# Function to calculate the gradients of the variables to be updated using the RK4 method
def gradient(vx, vy, D, mass):
    """
    Determines the gradients of the x and y displacements, as well as the gradients of the horizontal and vertical components of velocity.

    Args:
        vx (ndarray): Current horizontal components of velocity in ms^-1
        vy (ndarray): Current vertical components of velocity in ms^-1
        D (float): The drag parameter
        mass (float): Mass of the projectile in kg

    Returns:
        tuple: (dx/dt, dy/dt, dvx/dt, dvy/dt) the gradients of the displacements (horizontal and vertical) as well as the gradients of the velocity (horizontal and vertical)
    """

    # Differentiating displacement gives velocity
    dxdt = vx
    dydt = vy

    resultant_v = np.sqrt((vx ** 2) + (vy ** 2))

    # Calculates horizontal and vertical forces acting on the projectile
    fx = -D * resultant_v * vx
    fy = (mass * -G) - D * resultant_v * vy

    # Differentiating velocity to get acceleration
    dvxdt = fx / mass
    dvydt = fy / mass

    return np.vstack([dxdt, dydt, dvxdt, dvydt])


# Function to update important variables using the RK4 method
def RK4(vx, vy, x, y, D, mass, dt):
    """
    Calculate next displacements and velocities based on the RK4 method.

    Args:
        vx (ndarray): Current horizontal components of velocity in ms^-1
        vy (ndarray): Current vertical components of velocity in ms^-1
        x (ndarray): Current horizontal displacements in metres
        y (ndarray): Current vertical displacement in metres

    Returns:
        tuple: (x, y, vx, vy) the updated components of displacement and velocity (horizontal, vertical respectively)
    """

    # Assigning the gradients of the variables and calculating their k1, k2, k3 and k4 values based on the equations for each k value
    k1 = gradient(vx, vy, D, mass)
    k1 *= dt

    k2 = gradient(vx + (k1[2] / 2), vy + (k1[3] / 2), D, mass)
    k2 *= dt

    k3 = gradient(vx + (k2[2] / 2), vy + (k2[3] / 2), D, mass)
    k3 *= dt

    k4 = gradient(vx + k3[2], vy + k3[3], D, mass)
    k4 *= dt

    # Calculating the updated displacements and velocities by taking the weighted average of all the k values for each variable
    x = x + ((k1[0] + (2 * k2[0]) + (2 * k3[0]) + k4[0]) * (1 / 6))
    y = y + ((k1[1] + (2 * k2[1]) + (2 * k3[1]) + k4[1]) * (1 / 6))
    vx = vx + ((k1[2] + (2 * k2[2]) + (2 * k3[2]) + k4[2]) * (1 / 6))
    vy = vy + ((k1[3] + (2 * k2[3]) + (2 * k3[3]) + k4[3]) * (1 / 6))

    return (x, y, vx, vy)


# Main simulation function
def projectile_trajectory(velocities, angles, D, mass, dt, N):
    """
    Simulate projectile motion with air resistance using the Runge-Kutta 4th order method using vectorized operations.

    Args:
        velocities (ndarray): Launch speed in ms^-1
        angles (ndarray): Launch angle in radians
        D (float): Drag factor
        mass (float): Mass of the projectile in kg
        dt (float): Time step
    """

    x, y, t = np.zeros(N), np.zeros(N), np.zeros(N)
    angles = np.deg2rad(angles)

    horizontal_displacements = []
    vertical_displacements = []

    horizontal_velocities = velocities * np.cos(angles)
    vertical_velocities = velocities * np.sin(angles)

    # Tracking arrays as not all projectiles will hit the ground at the same time
    active = np.ones(N, dtype=bool)
    max_heights = np.zeros(N)
    ranges = np.zeros(N)

    while np.any(active):

        x_old, y_old = x.copy(), y.copy()

        # Calculate new position and velocity components
        x, y, horizontal_velocities, vertical_velocities = RK4(horizontal_velocities, vertical_velocities, x, y, D, mass, dt)

        horizontal_displacements.append(x.copy())
        vertical_displacements.append(y.copy())

        max_heights = np.maximum(max_heights, y)
        just_landed = (y <= 0) & active

        if np.any(just_landed):

            r = y_old[just_landed] / (y_old[just_landed] - y[just_landed])

            ranges[just_landed] = x_old[just_landed] + ((x[just_landed] - x_old[just_landed]) * r)

            y[just_landed] = 0.0
            x[just_landed] = ranges[just_landed]
            horizontal_velocities[just_landed] = 0.0
            vertical_velocities[just_landed] = 0.0

            active[just_landed] = False

    return ranges, max_heights