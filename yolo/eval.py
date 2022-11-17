import argparse
import random
import torch
from torchvision import datasets, transforms

from model import NeuralNetwork

DEFAULT_MODEL_PATH = 'weights/model34.pth'
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    data_transorm = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    test_data = datasets.CIFAR10(root='../../datasets/', train=False, download=True, transform=data_transorm["test"])
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
    model = NeuralNetwork(pretrained=False).to(opt.device)
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
