# Code of Pytorch

This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.

Reference: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

## Outline table
  |              | keep grad  | Backward | Foreward | loss function | update weights | 
  |:----:        |  :----:    |:----:          |:----:        |:----:    |:----:     |
  | NNinnumpy.py | X   | manually       | manually     | manually |manually   |


## NNinnumpy.py: Numpy

利用Numpy实现一个两层神经网络，网络结构为 "linear-Relu-linear-mean squared error".


## NNwithTensor.py: Tensors

与NNinnumpy.py不同点: 将np.array数据结构替换为tensor.

- Tensor与ndarray的区别：
  - Tensor支持GPU计算，而array不支持.
  - ***Tensors can keep track of a computational graph and gradients.***
- 指定计算设备：

       device = torch.device("cpu")
       device = torch.device("cuda:0")
- Variable, tensor, ndarray互相转换
       
       t = torch.from_numpy(n)  #ndarray 2 tensor
       n = t.numpy()  #tensor 2 ndarray
       
       v = Variable(t)  #tensor 2 variable
       
       v = Variable(torch.from_numpy(n))  #ndarray 2 variable
       n = v.numpy() #variable 2 ndarray



## NNwithAutograd.py： Tensors and autograd

与NNwithTenor.py的不同点：NNwithTensor.py是手动计算神经网络的前后传播，而NNwithAutograd是自动计算。

- Pytorch中的**autograd**包能够提供自主求导的功能（**automatic differentation**）

  - 前向传播定义为计算图(**computational graph**)，图中每个节点为Tensor，每个边为计算操作
  
  - 后向传播由autograd自动计算
  
- 如何计算某个tensor的梯度：

      x.requires_grad=True 
      w1 = torch.randn(D_in, H, requires_grad = True)
      
- loss.backward():本质是针对一个batch的输入数据来计算各个tensor的梯度


## NNwithnnmodel.py: nn module
与NNwithAutograd.py不同点：NNwithAutograd中的NN是用tensor矩阵计算实现的，NNwithnnmodel是利用nn package来构造神经网络

- The nn model only defines the structure, input and output, **without** the loss function and optimizer. Loss function and optimizer are both functions.

      loss_fn = torch.nn.MSELoss() #loss_fn为nn module 中定义的损失函数类的实体
      ...
      loss = loss_fn(y_pred,y)  
      ...
      loss.backward()

  - 输入为(y_pred, y_true)，输出loss.item()为loss值，需要在model之外定义
  - loss为一个tensor,但是loss保留了网络的整个计算图，可以供*.backward()*进行反向传播计算梯度，这也就是tensor flow的概念
  - loss.backward()可以直接更新在loss的tensor的梯度，但是并没有更新每个tensor的值
  
## NNwithoptim.py: torch.optim package

与NNwithnnmodel.py不同点：NNwithAutograd利用手动更新NN的参数的，而NNwithoptim.py利用optimizer来自动更新参数。

      optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

- optimizer根据tensor的值，梯度，历史梯度来更新网络的参数
  - 定义optimizer的时候需要指定需要**优化器的类型，待优化的参数(tensors)，学习率**
  - optimizer是针对每个batch的数据进行一次参数更新的，所以在for循环下执行该语句
  - 在每次循环中需要将optimizer的导数归零，否则会累积
  - optimizer更新一次参数：optimizer.step()
  - pytorch 和 keras在语句上一个重要的差别是，pytorch需要自己构造for循环，而keras有参数来指定循环次数。

## NNwithcustommodel: 

有很多时候我们想要构造特定结构的神经网络，此时我们可以继承**torch.nn.Module**定义自己的nn类：

      class TwoLayerNet(torch.nn.Module):
        def __init__():
          # 定义网络的结构，包括那些层，层节点的维数
        def forward():
          # 连接输入-层-输出，构造计算图

- 自定义神经网络类需要包括两个函数：**__init__()** 和 **forward()**
        

## 未完成：
- PyTorch: Defining new autograd functions
- TensorFlow: Static Graphs
- Pytorch: Control Flow + Weight sharing
  

