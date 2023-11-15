import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomHyperNet(nn.Module):
    def __init__(self, Nz, D, Nin, Nout, f):
        super(CustomHyperNet, self).__init__()
        self.Nin = Nin
        self.Nout = Nout
        self.f = f
        self.emb_dim = Nz 
        self.input_channels = Nin
        self.output_channels = Nout
        self.hidden_dense_layers = nn.ModuleList([nn.Linear(Nz, D) for _ in range(Nin)])
        self.final_layer = nn.Linear(D, Nout * f * f)

    def forward(self, emb):
        weights = []
        for layer in self.hidden_dense_layers:
            hidden = F.relu(layer(emb))
            weights.append(self.final_layer(hidden).unsqueeze(1))
        final_weights = torch.cat(weights, 1)
        return final_weights.view(-1, self.f, self.f, self.Nin, self.Nout)


class CustomHyperConvLayer(nn.Module):
    def __init__(self, hyper_net, input_channels, output_channels, kernel_size, padding='same', stride=1):
        super(CustomHyperConvLayer, self).__init__()
        self.hyper_net = hyper_net
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.emb_num = int((input_channels / hyper_net.input_channels) * (output_channels / hyper_net.output_channels))
        self.emb = nn.Parameter(torch.randn(self.emb_num, hyper_net.emb_dim))

    def forward(self, x):
        conv_weights = self.hyper_net(self.emb).view(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        return F.conv2d(x, conv_weights, stride=self.stride, padding=self.padding)
