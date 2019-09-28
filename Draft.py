# -*- coding: utf-8 -*-
"""
File: Draft.py
Project: CodeofPytorch
Author: Jiachen Zhao
Date: 9/16/19
Description:

Reference: https://cs230-stanford.github.io/pytorch-getting-started.html
"""
import numpy as np
import csv
import pandas as pd
import scipy.io as sio
import argparse #Parser for command-line options, argyments and sub-commands

#import packages from sklearn
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#import packages from torch
import torch
from torch.autograd import Variable
from torch import nn
from tensorboardX import SummaryWriter


class MatDataset(torch.utils.data.Dataset):
    """Build a data set from mat file
       The best advantage of dataloader is to save the memory space
    """
    def __init__(self, mat_file, transform=None):
        """
        :param mat_file: Path to the mat file with original data
        :param transform: OPtional transform to be applied on a sample
        """
        self.mat = sio.loadmat(mat_file)
        self.X = self.mat['X'].astype(float)
        self.y = self.mat['y']
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        feature = self.X[item,:]
        label = self.y[:,item]
        sample = {'X':feature, 'y': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Model(torch.nn.Module):
    def __init__(self,latent_dim):
        self.input_dim = 2
        self.latent_dim = latent_dim
        self.output_dim = 2
        super(Model, self).__init__()
        layers=[]
        layers += [nn.Linear(self.input_dim, self.latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(self.latent_dim, self.output_dim)]
        self.Net = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.Net(x)
        return outputs

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        # set the config
        self.__dict__.update(Solver.DEFAULTS, **config)
        # build data loader
        self.build_data()
        # build
        self.build_model()

    def build_data(self):
        # https://blog.csdn.net/VictoriaW/article/details/72356453
        # https://zhuanlan.zhihu.com/p/30934236
        trainset = MatDataset(mat_file='./Data/training.mat')
        testset = MatDataset(mat_file='./Data/testing.mat')
        self.num_train = len(trainset)
        self.num_test = len(testset)

        # step1.2 build dataloader based on dataset class
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config.batch_size, shuffle=False)
        # return num_train, num_test, train_loader, test_loader

    # build the model
    def build_model(self, verbose=False):

        self.model = Model(self.latent_dim)
        if verbose==True:
            print(model)
        self.cost = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # return model, cost, optimizer


    def draw_model(self):
        # how to download and build the tensorboard environment:
        # git clone https: // github.com / lanpa / tensorboardX & & cd tensorboardX & & python setup.py install
        logpath = 'log'
        dummy_input = torch.rand(10,2)# 假设输入10个3维样本
        with SummaryWriter(comment='draftNN') as w:
            w.add_graph(self.model, (dummy_input,))
        print('Draw the model structure in tensor board...')
        print('Tips: type the following code in Terminal of Pycharm (PATH is the path of runs in Project window.):\n'
              'tensorboard --logdir= PATH')

    # training
    def train(self):
        for epoch in range(self.num_epochs):
            # training stage
            running_loss = 0.0
            running_accuracy = 0.0
            print("Epoch {}/{}".format(epoch, self.num_epochs))
            for step, sample in enumerate(self.train_loader):
                X_train_epoch, y_train_epoch = sample['X'], sample['y']
                X_train_epoch, y_train_epoch = Variable(X_train_epoch), Variable(y_train_epoch).squeeze(1)
                outputs = self.model(X_train_epoch.float())

                self.optimizer.zero_grad()
                loss = self.cost(outputs, y_train_epoch)
                loss.backward()
                self.optimizer.step()
                _, pred = torch.max(outputs.data,1)
                running_loss += loss.item()
                running_accuracy += torch.sum(pred == y_train_epoch.data)
            print("Loss is:{:.8f}, Train Accuracy is:{:.2f}%".format(running_loss/self.num_train,
                  100*running_accuracy/self.num_train))

            # Validation
            testing_correct = 0
            for sample in self.test_loader:
                X_test_epoch, y_test_epoch = sample['X'], sample['y']
                X_test_epoch, y_test_epoch = Variable(X_test_epoch), Variable(y_test_epoch).squeeze(1)
                outputs = self.model(X_test_epoch.float())
                _, pred = torch.max(outputs.data, 1)
                testing_correct += torch.sum(pred == y_test_epoch.data)
            print("Test Accuracy is:{:.2f}%".format(100*testing_correct/self.num_test))

if __name__ == "__main__":
    print('--------')

parser = argparse.ArgumentParser()

# Model hyper-parameters
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--latent_dim', type=int, default=20)


# Training settings
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size',type=int, default=100)

config = parser.parse_args()


solver =Solver(vars(config))
solver.draw_model()
# solver.train()


# Extended content

# 1. 从.mat和.csv中读进数据,完成
# 2. 利用生成器来处理数据集， 完成
# 参看网络的结构,完成,print(model)
# 3. 自定义网络结构，而不使用sequential,完成
# 测量运行时间并保存
# 利用编译器来设置参数, 完成
# 保留每个阶段的模型
# 保存和加载模型
# 保存训练过程的损失函数，画训练函数曲线
# 4. dataset 类的transform
# 将历史数据保存到logger
# 如何设置dataloader的dtype
# customized loss function
# 利用solver来包装整个程序.优点是什么？, 完成
# 在tensorboard中可视化网络









