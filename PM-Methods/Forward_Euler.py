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

    drag_parameter = 0.5 * drag_coefficient * area * AIR_DENSITY
    return drag_parameter


# Main simulation function
def projectile_trajectory(initial_velocity, initial_angle, D, mass, dt):
    """
    Simulate projectile motion with air resistance using the Forward Euler method.

    Args:
        initial_velocity (float): Launch speed in ms^-1
        initial_angle (float): Launch angle in degrees
        D (float): Drag parameter
        mass (float): Mass of the projectile in kg
        dt (float): Time step in seconds

    Returns:
        tuple(horizontal displacements, vertical displacements) showing the x and y positions of the projectile at each interval
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

        # Calculate new position
        resultant_v = np.sqrt((horizontal_velocity ** 2) + (vertical_velocity ** 2))
        Force_x = -D * resultant_v * horizontal_velocity
        Force_y = (-mass * G) - D * resultant_v * vertical_velocity
        x = x + (dt * horizontal_velocity)
        horizontal_velocity = horizontal_velocity + (dt * (Force_x / mass))
        y = y + (dt * vertical_velocity)
        vertical_velocity = vertical_velocity + (dt * (Force_y / mass))

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
    drag_parameter = drag_parameter(drag_coefficient, area)

    # Running simulation
    displacements = projectile_trajectory(initial_velocity, initial_angle, drag_parameter, mass, dt)
    xs = displacements[0]
    ys = displacements[1]
    print(f"Range (3dps): {round(max(xs), 3)}, Maximum height (3dps): {round(max(ys), 3)}")

    # Plotting data
    plt.plot(xs, ys)
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.title("Projectile Trajectory")
    plt.show()