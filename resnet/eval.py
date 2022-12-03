import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import NeuralNetwork

DEFAULT_MODEL_PATH = 'weights/model.pth'
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    test_data = datasets.ImageFolder(root=os.path.join('../../datasets/hymenoptera_data', 'val'), transform=data_transform['val'])
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size, num_workers=4)

    x, y = next(iter(test_dataloader))
    x, y = x.to(opt.device), y.to(opt.device)
    return x, y


def show(images, labels):
    """
    展示图片
    :param images: 图片 images
    :param labels: 标签 list
    :return:
    """
    # 预训练时标准化的均值和方差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 对输入tensor进行处理
    images = [np.clip(np.transpose(image.cpu(), (1, 2, 0)), 0.0, 1.0) * std + mean for image in images]
    labels = [str(label) for label in labels]
    # 绘图
    n = int(np.ceil(np.sqrt(len(images))))
    fig, axes = plt.subplots(nrows=n, ncols=n, squeeze=False)
    for i in range(n * n):
        # 绘制子图
        if i < len(images):
            axes[i // n, i % n].imshow(images[i])
            axes[i // n, i % n].set_title(labels[i])
        # 去除边框
        for item in ['left', 'right', 'bottom', 'top']:
            axes[i // n, i % n].spines[item].set_visible(False)
        axes[i // n, i % n].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('result.png')
    # plt.show()


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model data path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    x, y = get_test_data(opt)
    # 模型
    classes = ['ants', 'bees']
    num_classes = len(classes)
    model = NeuralNetwork(num_classes).to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)
        predict = np.array([classes[i] for i in pred.argmax(dim=1)])  # 预测值
        actual = np.array([classes[i] for i in y])  # 真实值

        labels = [f'{predict[i]}' if predict[i] == actual[i] else f'{predict[i]}({actual[i]})'
                  for i in range(len(predict))]
        print(f'Accuracy: {100 * np.sum(predict == actual) / len(predict)}%.')
        show(x, labels)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
