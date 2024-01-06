# TTool

A toolkit for quick usage of PyTorch

## Clone

```bash
git clone https://github.com/IamK77/TTool.git
cd TTool
```

## Example

```python
import torchvision
import torchvision.transforms as transforms

from utils import TNet, CIFA10Net


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                download=False, transform=transform)

if __name__ == "__main__":

    tnet = TNet(CIFA10Net(), train_data=train_data, test_train=test_data, is_checkpoint=True)
    tnet.train()

    tnet.test('./target/best.pth')

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    TNet.infer('./infer/test_air.jpg', './target/best.pth', CIFA10Net(), 32, (1, 3, 32, 32), classes=classes)
```

## utils

`TNet`: Tool classes for training, testing, and inference.

`CIFA10Net`: Network architecture for CIFA10

`MnistNet`: Network architecture for Mnist

`nncheck`: Decorator for checking network architecture

## Support for custom network architecture validation

```python
import torch.nn as nn

@nncheck((1, 32, 32))
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Conv2d(1, 32, 5, 1, 2),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
    )

    def forward(self, x):
        x = self.model(x)
        return x
```

## Script

Install cuda support

```bash
curl -O https://raw.githubusercontent.com/IamK77/TTool/main/Scripts/cuda.bat && cmd /c cuda.bat ur_env_name
```

plz replace `ur_env_name` with your env name