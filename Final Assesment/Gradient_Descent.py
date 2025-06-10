import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise

# Add a bias(intercept) term to X
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(X)
    theta = np.random.randn(2, 1) # random initialization

    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients

    return theta

# Run gradient descent
theta_best = gradient_descent(X_b, y)

# Display the result
print(f"Learned parameters (theta):\n{theta_best}")

# Predict using the model
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add bias term
y_predict = X_new_b.dot(theta_best)

# Plotting
plt.plot(X_new, y_predict, "r-", label="Prediction")
plt.plot(X, y, "b.", label="Training data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression with Gradient Descent")
plt.show()