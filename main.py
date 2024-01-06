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