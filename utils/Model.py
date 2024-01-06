import os

from PIL import Image
import torch
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


from torch import nn
from torch.utils.data import DataLoader


class TNet():

    def __init__(self, net, train_data, test_train, epoch: int = 15, is_checkpoint: bool = False, is_summary: bool = False, only_best: bool = True) -> None:
        """
        :decr 初始化
        :param0 net 网络模型
        :param1 train_data 训练数据集
        :param2 test_data 测试数据集
        :param3 epoch 训练次数
        :param4 is_checkpoint 是否保存模型
        :param5 is_summary 是否使用 tensorboard
        :param6 only_best 是否只保存最好的模型
        """
        self.is_checkpoint = is_checkpoint
        self.checkpoint: bool = False
        self.checkepoch: int = 0
        self.epoch = epoch
        self.only_best = only_best
        if self.only_best:
            self.best = 0
            self.best_model = None
        self.is_summary = is_summary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_data = train_data
        self.test_data = test_train
        
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)

        print(f"训练数据集的长度为：{self.train_data_size}")
        print(f"测试数据集的长度为：{self.test_data_size}")

        batch_size = 64
        # 利用 DataLoader 来加载数据集
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
        sample, _ = self.train_data[0]
        channels, height, width = sample.shape
        print(f"Data size: {channels}, {height}, {width}")
        self.train_obj = net
        self.train_obj.to(self.device)
        print(summary(self.train_obj, sample.shape))

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)

        # 优化器
        self.learning_rate = 1e-2
        self.optimizer = torch.optim.SGD(self.train_obj.parameters(), lr=self.learning_rate, momentum=0.9)

        # 设置训练网络的一些参数
        # 记录训练的次数
        self.total_train_step = 0
        # 记录测试的次数
        self.total_test_step = 0

    def back_checkpoint_dict(self) -> dict:
        if self.is_checkpoint:
            return {
                'epoch': self.epoch,
                'model': self.train_obj.state_dict(),
                'point': True,
                'train_step': self.total_train_step,
                'test_step': self.total_test_step,
                'optimizer': self.optimizer.state_dict(),
            }
        else:
            return {
                'epoch': self.epoch,
                'model': self.train_obj.state_dict(),
            }

    def train(self):
        # 添加tensorboard
        if self.is_summary:
            writer = SummaryWriter("./logs_train")
        

        i = 1 if not self.checkpoint else self.checkepoch + 1

        while i <= self.epoch:
            print(f"-------第 {i} 轮训练开始-------")

            # 训练步骤开始
            self.train_obj.train()
            for data in self.train_dataloader:
                imgs, targets = data
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.train_obj(imgs)
                loss = self.loss_fn(outputs, targets)

                # 优化器优化模型
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.total_train_step = self.total_train_step + 1
                if self.total_train_step % 100 == 0:
                    print(f"训练次数：{self.total_train_step}, Loss: {round(loss.item(), 4)}")
                    if self.is_summary:
                        writer.add_scalar("train_loss", loss.item(), self.total_train_step)

            # 测试步骤开始
            self.train_obj.eval()
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for data in self.test_dataloader:
                    imgs, targets = data
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.train_obj(imgs)
                    loss = self.loss_fn(outputs, targets)
                    total_test_loss = total_test_loss + loss.item()
                    accuracy = (outputs.argmax(1) == targets).sum()
                    total_accuracy = total_accuracy + accuracy

            print(f"整体测试集上的Loss: {round(total_test_loss, 4)}")
            print(f"整体测试集上的正确率: {round((total_accuracy / self.test_data_size).item(), 4)}")

            if self.only_best:
                if (total_accuracy / self.test_data_size).item() > self.best:
                    self.best = (total_accuracy / self.test_data_size).item()
                    self.best_model = self.back_checkpoint_dict()

            if self.is_summary:
                writer.add_scalar("test_loss", total_test_loss, self.total_test_step)
                writer.add_scalar("test_accuracy", total_accuracy / self.test_data_size, self.total_test_step)

            self.total_test_step = self.total_test_step + 1

            if not os.path.exists('./target'):
                os.makedirs('./target')

            if not self.only_best:      
                torch.save(self.back_checkpoint_dict(), f"./target/train_{i}.pth")
        
            i = i + 1

        if self.only_best:
            torch.save(self.best_model, f"./target/best.pth")
            print(f"最好的模型已保存，正确率为：{self.best}")

        if self.is_summary:
            writer.close()

    
    def test(self, path):
        # 加载模型
        model = torch.load(path)
        self.train_obj.load_state_dict(model['model'])
        self.train_obj.to(self.device)
        self.train_obj.eval()

        total_test_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for data in self.test_dataloader:
                imgs, targets = data
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                # 进行推理
                outputs = self.train_obj(imgs)

                # 计算损失
                loss = self.loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()

                # 计算正确率
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print(f"测试集上的Loss: {round(total_test_loss, 4)}")
        print(f"测试集上的正确率: {round((total_accuracy / self.test_data_size).item(), 4)}")

    @staticmethod
    def infer(img_path, model_path ,net, img_size, input_shape: tuple, classes: dict = None) -> None:
        """
        :decr 推理
        :param1 img_path: 图像路径
        :param2 model_path: 模型路径
        :param3 net: 网络模型
        :param4 img_size: 图像大小
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = torch.load(model_path)
        net.load_state_dict(model['model'])
        net = net.to(device)
        net.eval()

        img = Image.open(img_path).resize((img_size, img_size))
        # img.show()
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img = transform(img)
        img = torch.reshape(img, input_shape)   # (1, 3, 32, 32)
        img = img.to(device)

        with torch.no_grad():
            output = net(img)

        _, predicted = torch.max(output.data, 1)
        if classes is None:
            print('Predicted: ', predicted.item())
        else:
            print('Predicted: ', classes[predicted.item()])


    def load_checkpoint(self, model_path):
        """
        Load model checkpoint.

        Parameters:
        model_path (str): The path to the saved model
        """
        checkpoint = torch.load(model_path)
        self.train_obj.load_state_dict(checkpoint['model'])
        # self.train_obj.eval()
        self.checkepoch = checkpoint['epoch']
        self.checkpoint = checkpoint['point']
        self.total_train_step = checkpoint['train_step']
        self.total_test_step = checkpoint['test_step']
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        print(f"模型已加载，当前训练次数为：{self.checkepoch}")
        self.train_obj.to(self.device)
        self.train_obj.train()

        self.train()
        

if __name__ == "__main__":
    from Net import CIFA10Net
    import torchvision

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                    download=False, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                    download=False, transform=transform)
    net = TNet(CIFA10Net(), train_data=train_data, test_train=test_data, is_checkpoint=True, epoch=20)
    net.train()
    net.test('./target/train_18.pth')
    net.load_checkpoint('./target/train_15.pth')
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    TNet.infer('./infer/test_air.jpg', './target/train_18.pth', CIFA10Net(), 32, (1, 3, 32, 32), classes=classes)