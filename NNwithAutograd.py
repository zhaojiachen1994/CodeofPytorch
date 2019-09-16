# -*- coding: utf-8 -*-
"""
File: NNwithAutograd.py
Project: CodeofPytorch
Author: Jiachen Zhao
Date: 9/16/19
Description:
"""
import torch

# Define the dtype and device
dtype = torch.float
device = torch.device("cpu")

# Define the dimensions of tensors
N, D_in, H, D_out = 64, 1000, 100, 10

# Randomly initialize x, y, w1, w2
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):    #500 is # of iterations
    # Forward pass: define the computational graph
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute the loss function
    loss = (y_pred-y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item()) # loss.item() gets the scalar value held in the loss.

    # Backward pass: compute the gradients.
    loss.backward()

    # Manually update weights using gradient descent.
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        # w -= 0.01 * w.grad is an in-place operation,
        # so it performs calculation on existing w and updates the value.
        # However, w = w - 0.01 * w.grad is not in-place operation,
        # so it creates a new variable w, which does not have requires_grad set and so the error.

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()