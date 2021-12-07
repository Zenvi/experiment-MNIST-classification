import torch
import torch.nn as nn
import torch.nn.functional as F

'''My ConvNet in PyTorch.'''
class my_FCNet_0(nn.Module):
    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(1 * 28 * 28, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.network(x)
        x = x.squeeze()
        return x

class my_ConvNet_1(nn.Module):
    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.network(x)
        x = x.squeeze()
        return x

class my_ConvNet_2(nn.Module):
    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0),

            nn.AdaptiveAvgPool2d(output_size=1)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.squeeze()
        return x

def test():
    net = my_ConvNet_1()
    print(net)
    x = torch.randn(1,1,28,28)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
