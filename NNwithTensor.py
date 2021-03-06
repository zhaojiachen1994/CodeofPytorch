# -*- coding: utf-8 -*-
"""
File: NNwithTensor.py
Project: CodeofPytorch
Author: Jiachen Zhao
Date: 9/16/19
Description: Referred from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)#Randomly initialize the input, point the device and dtype
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H,D_out, device=device, dtype=dtype)



learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)# x.dot(w1), mm for matrix multiply
    h_relu = h.clamp(min=0)# np.maximum(0,h) torch.clamp(input, min, max) 将输入input张量每个元素的夹紧到区间 [min,max]
    y_pred = h_relu.mm(w2)# h_relu.dot(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).mean().item() # np.square(y_pred - y).mean()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)# h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t()) # grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.clone() # grad_h_relu.copy(), Real copy in pytorch: .clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2