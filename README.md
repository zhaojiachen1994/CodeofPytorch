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

-Pytorch中的**autograd**包能够提供自主求导的功能（*automatic differentation*）
