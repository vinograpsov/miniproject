import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from HyperNet import CustomHyperNet, CustomHyperConvLayer


class MNISTHyperNetModel(nn.Module):
    def __init__(self, emb_dim=4, emb_proj_dim=4, filter_size=7, num_class=10):
        super(MNISTHyperNetModel, self).__init__()

        self.hnet = CustomHyperNet(Nz=emb_dim, D=emb_proj_dim, Nin=1, Nout=16, f=filter_size)
        self.hnet2 = CustomHyperNet(Nz=emb_dim, D=emb_proj_dim, Nin=16, Nout=16, f=filter_size)

        self.conv1_hlayer = CustomHyperConvLayer(hyper_net=self.hnet, input_channels=1, output_channels=16, kernel_size=filter_size, padding='same')
        self.max1_layer = nn.MaxPool2d(kernel_size=2)

        self.conv2_hlayer = CustomHyperConvLayer(hyper_net=self.hnet2, input_channels=16, output_channels=16, kernel_size=filter_size, padding='same')
        self.max2_layer = nn.MaxPool2d(kernel_size=2)

        self.flatten_layer = nn.Flatten()
        self.final_layer = nn.Linear(16 * 7 * 7, num_class) 

    def forward(self, x):
        x = F.relu(self.conv1_hlayer(x))
        x = self.max1_layer(x)
        x = F.relu(self.conv2_hlayer(x))
        x = self.max2_layer(x)
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return F.log_softmax(x, dim=1)


class StandardCNNModel(nn.Module):
    def __init__(self, filter_size=7, num_class=10):
        super(StandardCNNModel, self).__init__()
        self.conv1_layer = nn.Conv2d(1, 16, kernel_size=filter_size, padding='same')
        self.max1_layer = nn.MaxPool2d(kernel_size=2)
        self.conv2_layer = nn.Conv2d(16, 16, kernel_size=filter_size, padding='same')
        self.max2_layer = nn.MaxPool2d(kernel_size=2)
        self.flatten_layer = nn.Flatten()
        self.final_layer = nn.Linear(16 * 7 * 7, num_class)

    def forward(self, x):
        x = F.relu(self.conv1_layer(x))
        x = self.max1_layer(x)
        x = F.relu(self.conv2_layer(x))
        x = self.max2_layer(x)
        x = self.flatten_layer(x)
        x = self.final_layer(x)
        return F.log_softmax(x, dim=1)
