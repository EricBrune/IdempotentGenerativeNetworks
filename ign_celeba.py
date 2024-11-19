import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # 64->32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # 32->16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         # 16->8
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)         # 8->4
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0)         # 4->1

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0)  # 1->4
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 4->8
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8->16
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 16->32
        self.convt5 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 32->64

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.convt2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.convt3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.convt4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.convt5(x)
        x = self.tanh(x)
        return x

class IGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)