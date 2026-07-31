from PM_Methods.vectorized_RK4 import drag_parameter

# Physics simulation parameters
dt = 0.0001
drag_coefficient = 0.51  # Standard padel ball drag coefficient used by testing facilities such as the Tennis Warehouse Racquet Analyzer
area = 0.0135333  # Average area of FIP approved padel balls in m^2
mass = 0.0577  # Average FIP approved mass of a padel ball in kg
D = drag_parameter(drag_coefficient, area)

# Machine learning parameters
N = 500 # Number of simulated projectiles and number of epochs for the MLP
learning_rate = 0.001
