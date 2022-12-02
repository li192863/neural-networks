import argparse
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

import utils
from dataset import PennFudanDataset
from train import get_transform
from model import get_model_instance_segmentation

NUM_CLASSES = 2  # background(0) and person(1)
DATASET_ROOT_PATH = '../../datasets/PennFudanPed/'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = PennFudanDataset(DATASET_ROOT_PATH, get_transform(train=False))
    idx = random.randint(0, len(test_data) - 1)
    x, y = test_data[idx]  # 随机获取测试数据
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2]).to(opt.device)
    return x, y


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model weights path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def show(imgs):
    """
    展示图片
    :param imgs:
    :return:
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('result.png')
    # plt.show()


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
        proba_threshold = 0.5
        score_threshold = 0.75
        masks = [out['masks'][out['scores'] > score_threshold] > proba_threshold for out in pred]

        origin = x.squeeze().mul(255).byte().cpu()  # tensor, shape: (3, h, w), dtype: uint8, device: cpu
        output = draw_segmentation_masks(origin, masks[0].squeeze(), alpha=0.9)
        show([origin, output])
        print(pred[0]['scores'])

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
