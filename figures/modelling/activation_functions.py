import numpy as np
import matplotlib.pyplot as plt

# Define the step function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Define the sigmoid function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Define the ReLU function
def relu_function(x):
    return np.maximum(0, x)

# Generate x values
x = np.linspace(-6, 6, 500)

# Compute y values for all functions
y_step = step_function(x)
y_sigmoid = sigmoid_function(x)
y_relu = relu_function(x)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y_step, label="Step Function", color="black")
plt.plot(x, y_sigmoid, linestyle="dotted", label="Sigmoid Function", color="black")
plt.plot(x, y_relu, linestyle="dashdot", label="ReLU Function", color="black")

# Add labels, legend, and grid
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

# Set y-ticks to -1, 0, and 1
plt.yticks([0, 0.5, 1])
plt.ylim(-0.1, 1.1)

# Save the plot as a PNG file
plt.savefig("activation_functions.png", dpi=300, bbox_inches="tight")