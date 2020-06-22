# import tensorflow as tf
# from tensorflow import keras

# import matplotlib.pyplot as plt


import os
from os import listdir

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


import Constants
from DataManager import DataManager


# def load_split_train_test(datadir, valid_size = .2):
#     train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
#     test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
#
#     train_data = datasets.ImageFolder(datadir, transform=train_transforms)
#     test_data = datasets.ImageFolder(datadir, transform=test_transforms)
#
#     num_train = len(train_data)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))
#     np.random.shuffle(indices)
#     train_idx, test_idx = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_idx)
#     test_sampler = SubsetRandomSampler(test_idx)
#     trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler,batch_size=64)
#     testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
#     return trainloader, testloader

def load_split_train_test(datamanager, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    # train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    # test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler,batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader

if __name__ == '__main__':
    d = DataManager(Constants.MINI_DATASET)
    print(type(d.dataset.writing_style_to_sessions), d.dataset.writing_style_to_sessions)
    # print(torch.__version__)
    # print(os.path.dirname(os.path.realpath(__file__)))
    # print(Constants.DATASET)
    # for user_folder in os.listdir(Constants.DATASET_DIRECTORY_PATH):
    #     print(user_folder)


    # for f in listdir(Constants.DATASET_DIRECTORY_PATH+Constants.MINI_DATASET):
    #     print(f)
    # trainloader, testloader = load_split_train_test(data_dir, .2)
    # print(trainloader.dataset.classes)


    # print("ciao")
    # print(torch.cuda.is_available())
    #
    # a = torch.arange(5*5).view(5, 5).float().cuda()
    # print(a)
    # b = (torch.rand(5, 5) > 0.5).float().cuda()
    # print(b)
    # c = a @ b
    # print(c)
    # print(a * b)


    # print(tf.__version__)
    # fashion_mnist = keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # print(type(train_images))