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
        drag_coefficient (float): The drag coefficient of the projectile, dependant on the shape of the projectile
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
        vx (float): Current horizontal component of velocity in ms^-1
        vy (float): Current vertical components of velocity in ms^-1
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

    return np.array([dxdt, dydt, dvxdt, dvydt])


# Function to update important variables using the RK4 method
def RK4(vx, vy, x, y, D, mass, dt):
    """
    Calculate next displacements and velocities based on the RK4 method.

    Args:
        vx (float): Current horizontal component of velocity in ms^-1
        vy (float): Current vertical component of velocity in ms^-1
        x (float): Current horizontal displacement in m
        y (float): Current vertical displacement in m

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
def projectile_trajectory(initial_velocity, initial_angle, D, mass, dt):
    """
    Simulate projectile motion with air resistance using the Runge-Kutta 4th order method.

    Args:
        initial_velocity (float): Launch speed in ms^-1
        initial_angle (float): Launch angle in radians
        D (float): Drag factor
        mass (float): Mass of the projectile in kg
        dt (float): Time step
    """

    x, y, t = 0, 0, 0
    initial_angle = np.deg2rad(initial_angle)

    horizontal_displacements = []
    vertical_displacements = []

    horizontal_velocity = initial_velocity * np.cos(initial_angle)
    vertical_velocity = initial_velocity * np.sin(initial_angle)

    while True:

        horizontal_displacements.append(x)
        vertical_displacements.append(y)

        x_old, y_old, t_old = x, y, t

        # Calculate new position and velocity components
        x, y, horizontal_velocity, vertical_velocity = RK4(horizontal_velocity, vertical_velocity, x, y, D, mass, dt)

        # Updating the time
        t += dt

        # Collision detected and correction
        if y < 0:
            r = y_old / (y_old - y)  # Fraction of the time step after which the projectile hit the ground
            x = x_old + ((x - x_old) * r)
            y = y_old + ((y - y_old) * r)
            t = t - ((1 - r) * dt)

            horizontal_displacements.append(x)
            vertical_displacements.append(y)

            return (horizontal_displacements, vertical_displacements)


if __name__ == "__main__":
    # Parameters
    initial_velocity = 30.0
    initial_angle = 45.0
    dt = 0.0001
    drag_coefficient = 0.51  # Standard padel ball drag coefficient used by testing facilities such as the Tennis Warehouse Racquet Analyzer
    area = 0.0135333  # Average area of FIP approved padel balls in m^2
    mass = 0.0577  # Average FIP approved mass of a padel ball in kg
    D = drag_parameter(drag_coefficient, area)

    # Running simulation
    displacements = projectile_trajectory(initial_velocity, initial_angle, D, mass, dt)
    xs = displacements[0]
    ys = displacements[1]
    print(f"Range (3dps): {round(max(xs), 3)}, Maximum height (3dps): {round(max(ys), 3)}")

    # Plotting data
    plt.plot(xs, ys)
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Trajectory")
    plt.show()