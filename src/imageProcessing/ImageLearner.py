# import tensorflow as tf
# from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

import Constants
from imageProcessing.ImageDataManager import DataManager


def load_split_train_test(datamanager, writing_style=Constants.ITALIC, valid_size = .2):
    datadir = Constants.DATASET_DIRECTORY_PATH + datamanager.dataset_name + Constants.FOR_TORCH_FOLDER_SUFFIX + '/' + writing_style + '/'
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])
    print("READING DATASETS...")
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    print("PREPROCESSING DATA...")
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler,batch_size=32, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=32, num_workers=0)
    return trainloader, testloader

def train_model(model, trainloader, testloader, epochs):
    print("TRAINING MODEL...")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)

    # epochs = 50
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [],  []

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            my_logps = model.forward(inputs)
            loss = criterion(my_logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        my_logps = model.forward(inputs)
                        batch_loss = criterion(my_logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(my_logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'aerialmodel.pth')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()



if __name__ == '__main__':
    # print(torch.cuda.is_available())

    d = DataManager(Constants.IMG_DATASET)
    trainloader, testloader = load_split_train_test(d)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model1 = models.resnet50(pretrained=True)
    # model = models.alexnet(pretrained=True, progress=True)
    # print(model)
    train_model(model1, trainloader, testloader, 30)
    # print(type(trainloader), type(testloader))
    # print(type(d.dataset.writing_style_to_sessions))
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