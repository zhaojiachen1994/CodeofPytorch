# -*- coding: utf-8 -*-
"""
File: NNwithnnmodule.py
Project: CodeofPytorch
Author: Jiachen Zhao
Date: 9/17/19
Description:
"""

import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N,D_out)

# Define the nn model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# Define the loss function
loss_fn = torch.nn.MSELoss()

learning_rate=1e-2
for t in range(500):
    # Forward pass and loss function
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    if t % 100 ==99:
        print(t, loss.item())

    # Set the parameter grad to 0
    model.zero_grad()

    # Backward pass
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad













