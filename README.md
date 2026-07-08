# Projectile Motion and Machine Learning

This project aims to accurately predict the maximum heights and ranges of projectiles given their initial velocities and
launch angles, while also building the underlying systems used to do this.

## Simulating projectile trajectory

In order to predict this information, the model requires training data. I simulated projectile trajectory (with air 
resistance) using numerical methods in two distinct ways. 

### (1) Forward Euler Method 

**What it does:** - Simulates projectile trajectory using a first order approximation method

**Key conepts** - Easy to code and understand but computationally expensive and can be numerically unstable and less 
accurate

Graph demonstrating an example projectile trajectory of a padel ball with an initial speed of 30m/s and launch angle of 
45 degrees:

![Forward Euler projectile trajectory graph](images/Forward-Euler-graph.png)
Note - To learn and implement the Forward Euler method, I studied the following paper by Tomasz Chwiej: 
https://galaxy.agh.edu.pl/~chwiej/comp_phys/labs/1_projectile_launch.pdf

### (2) Runge-Kutta 4 Method

**What it does** - Simulates projectile trajectory using a fourth order method approximation method

**Key concepts** - Much more accurate and numerically stable than the Forward Euler method while also being less 
computationally expensive

Graph for the trajectory of a padel ball with the same conditions but using the RK4 method:

![RK4 method projectile trajectory graph](images/RK4-graph.png)


## Building the Models

### (1) Linear Regression from scratch - Gradient Descent

**What it does** - Predicts the equation of a line given its 'x' and 'y' values using gradient descent

**Key concepts** - Can accurately predict linear equations but struggles with polynomial equations, requires fine-tuning
parameters such as learning rate, epochs and number of features and labels

Graph showing the model predicting the linear equation 'y = 3x + 4'

![Linear Regression Gradient Descent graph](images/linear-regression-gd-graph.png)
Notice that the predicted line overlaps almost perfectly with the straight line 'y'. 

Graph showing the model predicting the quadratic function 'y = x^2 + 4':

![Linear Regression Gradient Descent poly graph](images/LRquad.png)
Note - The quadratic is not smooth and only has x values between -10 and 10. Any more than 20 x values and the model
runs into an overflow error

### (2) Linear Regression from scratch - Normal equation

**What it does** - Predicts the equation of a line given the lines x and y values by calculating the exact weight in one
line

**Key concepts** - Perfectly predicts linear and polynomial equations and doesn't require fine-tuning any parameters - 
much more accurate than gradient descent especially for higher degree polynomials and doesn't require feature-scaling.

Graph showing the models prediction of 'y = 3x + 4':

![Linear Regression Normal Linear](images/normalLinear.png)

Graph showing the models prediction of 'y = x^2 + 4':

![Linear Regression Normal poly](images/Normalquad.png)
Notice how much smoother the curve is as compared to gradient descents version