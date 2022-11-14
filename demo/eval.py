import argparse
import random
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork

DEFAULT_MODEL_PATH = 'weights/model.pth'
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = datasets.FashionMNIST(root='../../datasets/', train=False, download=True, transform=ToTensor())
    x, y = test_data[random.randint(0, len(test_data) - 1)]  # 随机获取测试数据
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]).to(opt.device)
    return x, y


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model weights path')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    x, y = get_test_data(opt)
    # 模型
    model = NeuralNetwork().to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: \'{predicted}\', Actual: \'{actual}\'')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
