# Code of Pytorch
## NNinnumpy.py: Numpy

利用Numpy实现一个两层神经网络，网络结构为 "linear-Relu-linear-mean squared error".


## NNwithTensor.py: Tensors

与NNinnumpy.py不同点: 将np.array数据结构替换为tensor.

- Tensor与ndarray的区别：
  - Tensor支持GPU计算，而array不支持.
  - Tensors can keep track of a computational graph and gradients.
- 指定计算设备：

       device = torch.device("cpu")
       device = torch.device("cuda:0")
- Variable, tensor, ndarray互相转换
       
       t = torch.from_numpy(n)  #ndarray 2 tensor
       n = t.numpy()  #tensor 2 ndarray
       
       v = Variable(t)  #tensor 2 variable
       
       v = Variable(torch.from_numpy(n))  #ndarray 2 variable
       n = v.numpy() #variable 2 ndarray



## NNwithAutograd： Tensors and autograd

与NNwithTenor.py的不同点：NNwithTensor.py是手动计算神经网络的前后传播，而NNwithAutograd是自动计算。

- Pytorch中的**autograd**包能够提供自主求导的功能（**automatic differentation**）

  - 前向传播定义为计算图(**computational graph**)，图中每个节点为Tensor，每个边为计算操作
  
  - 后向传播由autograd自动计算
  
- 如何计算某个tensor的梯度：

      x.requires_grad=True 
      w1 = torch.randn(D_in, H, requires_grad = True)
      
- loss.backward():本质是针对一个batch的输入数据来计算各个tensor的梯度


## NNwithnnmodel: nn
与NNwithAutograd.py不同点：利用nn package来构造神经网络

- The nn model only defines the structure, input and output, **without** the loss function and optimizer. Loss function and optimizer are both functions.

- loss function 'loss_fn = torch.nn.MSELoss(reduction='sum')' 可以理解为nn module 中定义的损失函数类的实体，函数输入为(y_pred, y_true)，loss.item()为loss值。

## NNwith

