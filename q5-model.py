import torch
import torch.nn as nn
import torch.nn.functional as F

class SeBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class C3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()
        self.Se = SeBlock(out_channels)

    def forward(self, x):
        y = x
        y = self.res(y)
        x = self.conv(x)
        z = x
        z = self.Se(z)
        x = x * z + y
        return self.relu(x)


class Classification(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_1 = C3(3, 64)
        self.pool_1 = nn.AdaptiveMaxPool2d((64, 64))
        self.c_2 = C3(64, 128)
        self.pool_2 = nn.AdaptiveMaxPool2d((32, 32))
        self.c_3 = C3(128, 256)
        self.pool_3 = nn.AdaptiveMaxPool2d((16, 16))
        self.c_4 = C3(256, 512)
        self.pool_4 = nn.AdaptiveMaxPool2d((8, 8))
        self.c_5 = C3(512, 1024)
        self.pool_5 = nn.AdaptiveMaxPool2d((4, 4))
        self.c_6 = C3(1024, 1024)
        self.pool_6 = nn.AdaptiveMaxPool2d((1, 1))
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.out_channels),
        )

    def forward(self, x):
        x = self.c_1(x)
        x = self.pool_1(x)
        x = self.c_2(x)
        x = self.pool_2(x)
        x = self.c_3(x)
        x = self.pool_3(x)
        x = self.c_4(x)
        x = self.pool_4(x)
        x = self.c_5(x)
        x = self.pool_5(x)
        x = self.c_6(x)
        x = self.pool_6(x)
        x = self.flat(x)
        x = self.linear(x)
        return x