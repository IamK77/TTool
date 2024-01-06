import torch
from torchsummary import summary
from torch import nn


def nncheck(input_shape):
    def decorator(cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = cls().to(device)
        print(summary(net, input_shape))
        return cls
    return decorator

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Conv2d(1, 32, 5, 1, 2),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 5, 1, 2),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64*7*7, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 10)
    )

    def forward(self, x):
        x = self.model(x)
        return x
    

class CIFA10Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.Linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10),
        )

    def forward(self,x):
            x = self.Conv(x)
            x = self.Linear(x)
            return x
    
# @nncheck((1, 32, 32))
class Lenet_5(nn.Module):
     
    def __init__(self) -> None:
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.Linear = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self,x):
        x = self.Conv(x)
        x = self.Linear(x)
        return x


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Lenet_5().to(device)
    # print(summary(net, (1, 32, 32)))
    Lenet_5()
    




