import argparse
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFont, Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import NeuralNetwork

DEFAULT_MODEL_PATH = 'data/model.pth'
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


def show_classification_result(images, labels, image_size=None, text_color=None):
    """
    展示图片
    :param images: 图片 images
    :param labels: 标签 list
    :param image_size: 图片大小
    :param text_color: 文本颜色
    :return:
    """
    # 预处理图片
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))  # 预训练时标准化的均值
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))  # 预训练时标准化的方差
    images = [np.clip(image.cpu() * std + mean, 0.0, 1.0) for image in images]  # 对输入tensor进行处理
    labels = [str(label) for label in labels]  # 对输入tensor进行处理

    # 绘制每张图
    scale = 16  # 设置字体缩放大小
    image_size = image_size or images[0].shape[1:]  # 获取图片大小
    font = ImageFont.truetype(font='data/Microsoft YaHei.ttf', size=sum(image_size) // scale)  # 设置字体

    num_images = len(images)
    for i in range(num_images):
        # 转换为PIL图像
        image = F.to_pil_image(images[i])
        image = image.resize(image_size)  # 放大以更清楚显示
        # 绘制标题
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), labels[i], font=font, fill=text_color)
        # 转换为tensor
        images[i] = F.pil_to_tensor(image)

    # 生成网格图
    nrow = 8
    # nrow = int(np.ceil(np.sqrt(len(images))))
    result = make_grid(images, nrow=nrow)
    result = F.to_pil_image(result)
    result.save('result.png')
    result.show()


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
        show_classification_result(x, labels)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
