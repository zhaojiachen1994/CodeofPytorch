# Code of Pytorch
**NNinnumpy.py**

利用Numpy实现一个两层神经网络，网络结构为 "linear-Relu-linear-mean squared error".

---

**NNwithTensor.py**

与NNinnumpy.py不同点: 将np.array数据结构替换为tensor.

- Tensor与ndarray的区别：
  - Tensor支持GPU计算，而array不支持.
  - Tensors can keep track of a computational graph and gradients.
- 指定计算设备：

       device = torch.device("cpu")
       device = torch.device("cuda:0")

---


**NN**

