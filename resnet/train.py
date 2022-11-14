import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import NeuralNetwork

DATASET = datasets.CIFAR10
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'weights/model34.pth'


def get_dataloader(dataset, opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    training_data = dataset(root='../../datasets/', train=True, download=True, transform=data_transform['train'])
    test_data = dataset(root='../../datasets/', train=False, download=True, transform=data_transform['test'])
    train_dataloader = DataLoader(training_data, batch_size=opt.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer, opt):
    """
    训练模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param opt:
    :return:
    """
    model.train()  # Sets the module in training mode
    with tqdm(dataloader, desc=f'Epoch {opt.epoch}/{opt.epochs}, train', total=len(dataloader)) as pbar:  # 进度条
        for X, y in pbar:
            # 前向传播
            X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
            pred = model(X)  # 预测结果
            loss = loss_fn(pred, y)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 打印信息
            pbar.set_postfix({'loss': f'{loss.item():>7f}'})


def test(dataloader, model, loss_fn, opt):
    """
    测试模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param opt:
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader, desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test', total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                test_loss += loss_fn(pred, y).item()  # 计算损失
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 判断正误
                pbar.set_postfix({'Accuracy': f'{(100 * correct / size):>0.1f}%', 'Avg loss': f'{test_loss / num_batches:>8f}'})


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path', default=DEFAULT_SAVE_PATH, help='model save path')
    return parser.parse_args()


def main(opt):
    """
    主函数
    :param opt:
    :return:
    """
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    train_dataloader, test_dataloader = get_dataloader(DATASET, opt)
    # 模型
    model = NeuralNetwork().to(opt.device)
    # 参数
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters())  # 优化器
    # 训练
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train(train_dataloader, model, loss_fn, optimizer, opt)  # 训练
        test(test_dataloader, model, loss_fn, opt)  # 测试
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
