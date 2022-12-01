import argparse
import random
import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor

from dataset import PennFudanDataset
from mask_rcnn.train import get_transform
from model import get_model_instance_segmentation

NUM_CLASSES = 2  # background(0) and person(1)
DATASET_ROOT_PATH = '../../datasets/PennFudanPed/'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = PennFudanDataset(DATASET_ROOT_PATH, get_transform(train=False))
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
    model = get_model_instance_segmentation(NUM_CLASSES).to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)
        print(pred)
        input = Image.fromarray(x.squeeze().mul(255).permute(1, 2, 0).byte().cpu().numpy())
        input.show()
        for i in range(len(pred[0]['masks'])):
            output = Image.fromarray(pred[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
            output.show()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
