"""
NeuraNet - tiny beginner neural network
Run: python neuranet.py

This code uses only numpy and trains a tiny network on a simple problem.
Comments explain each step in plain language.
"""

import numpy as np

# ---------- Simple dataset (XOR-like toy)
# Inputs (4 examples, 2 features each)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Targets (we try to teach the network to output 1 when exactly one input is 1)
y = np.array([[0], [1], [1], [0]], dtype=float)

# ---------- Network size
input_size = 2   # two inputs
hidden_size = 4  # small hidden layer
output_size = 1  # single output

# Random seed so results are repeatable
np.random.seed(1)

# ---------- Initialize weights (small random numbers)
W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))

# ---------- Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# ---------- Training settings
lr = 0.5       # learning rate (how big steps are)
epochs = 10000  # number of training steps

# ---------- Training loop (very basic)
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1         # input to hidden (before activation)
    a1 = sigmoid(z1)                # hidden activation
    z2 = np.dot(a1, W2) + b2        # hidden to output (before activation)
    a2 = sigmoid(z2)                # network output

    # Compute simple mean squared error
    loss = np.mean((a2 - y) ** 2)

    # Print progress occasionally
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, loss: {loss:.4f}")

    # Backpropagation (compute gradients)
    d_a2 = 2 * (a2 - y) / y.size           # derivative of loss w.r.t output activation
    d_z2 = d_a2 * sigmoid_deriv(z2)       # output layer delta
    dW2 = np.dot(a1.T, d_z2)
    db2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_deriv(z1)
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    # Gradient descent parameter update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# ---------- After training show results
print("\nTraining finished.\nOutputs after training:")
preds = (a2 > 0.5).astype(int)
for i, x in enumerate(X):
    print(f"Input: {x} -> Pred: {preds[i][0]}, Prob: {a2[i][0]:.3f}, True: {int(y[i][0])}")
