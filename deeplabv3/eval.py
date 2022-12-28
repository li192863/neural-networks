import argparse
import colorsys
import math
from functools import reduce

import torch
import torchvision.transforms.functional as F
from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import DataLoader
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks, make_grid

import utils
from deeplabv3.dataset import MotorcycleNightRideDataset
from deeplabv3.model import get_model_sematic_segmentation
from deeplabv3.presets import SegmentationPresetEval

DATASET_ROOT_PATH = '../../datasets/Motorcycle Night Ride'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_BATCH_SIZE = 6
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_WORKERS = 16
classes = ['Undrivable', 'Road', 'Lanemark', 'My bike', 'Rider', 'Movable']


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = MotorcycleNightRideDataset(DATASET_ROOT_PATH, transforms=SegmentationPresetEval(base_size=520))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True,
                                                  num_workers=opt.workers, collate_fn=utils.collate_fn)

    x, y = next(iter(test_dataloader))
    x, y = x.to(opt.device), y.to(opt.device)
    return x, y


def show_sematic_segmentation_result(images, masks, labels, image_size=None, colors=None):
    """
    展示语义分割结果
    :param images:
    :param boxes:
    :param labels:
    :param image_size:
    :param colors:
    :return:
    """
    # 预处理图片
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape((3, 1, 1))  # 预训练时标准化的均值
    std = torch.tensor([0.229, 0.224, 0.225]).reshape((3, 1, 1))  # 预训练时标准化的方差
    images = [torch.as_tensor(torch.clip(image.cpu() * std + mean, 0.0, 1.0) * 255, dtype=torch.uint8) for image in images]  # 对输入tensor进行处理
    masks = [torch.as_tensor(mask.cpu()) for mask in masks]  # 对输入tensor进行处理

    # 裁切图片与蒙版
    images = [remove_edge(image) for image in images]
    masks = [mask[:, :image.shape[1], :image.shape[2]] for mask, image in zip(masks, images)]

    # 绘制每张图
    font = ImageFont.truetype(font='data/Microsoft YaHei.ttf')  # 设置字体
    num_images = len(images)
    image_size = image_size or (sum(list(map(lambda x: x.shape[2], images))) // num_images,
                                sum(list(map(lambda x: x.shape[1], images))) // num_images)  # 获取图片大小(W, H)
    for i in range(num_images):
        image = draw_segmentation_masks(images[i], masks[i], colors=colors)
        image = F.to_pil_image(image)

        mask = torch.sum(masks[i], dim=[1, 2]) != 0  # not empty mask
        boxes = masks_to_boxes(masks[i][mask])  # 转为边界框
        origin = torch.stack([(boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2])  # 获取边界框中心点
        box_labels = [item for i, item in enumerate(labels) if mask[i]]  # 获取边界框的标签
        draw = ImageDraw.Draw(image)
        for j in range(len(boxes)):
            draw.text(origin[:, j].tolist(), box_labels[j], font=font)
        image = letterbox_image(image, image_size)

        image = F.to_tensor(image)
        images[i] = image

    # 生成网格图
    # nrow = 8
    nrow = int(math.ceil(math.sqrt(len(images))))
    result = make_grid(images, nrow=nrow)
    result = F.to_pil_image(result)
    result.save('data/result.png')
    result.show()


def remove_edge(image):
    """
    去除图片右边以及下边外围多余黑边
    :param image:
    :return:
    """
    mask = image == image[:, -1, -1].reshape(3, 1, 1)  # equals last pixel mask
    mask = mask[0, ...] & mask[1, ...] & mask[2, ...]  # 深度方向逻辑与操作，得到[H, W]的张量
    line_h, line_w = reduce(lambda a, b: a & b, mask.transpose(1, 0)), reduce(lambda a, b: a & b, mask)  # 高度方向逻辑与操作
    idx_h, idx_w = torch.where(line_h)[0], torch.where(line_w)[0]  # 获取值为true的下标
    max_h = torch.min(idx_h) if len(idx_h) != 0 else image.size(1)  # 获取图片边界对应的下标max_h
    max_w = torch.min(idx_w) if len(idx_w) != 0 else image.size(2)  # 获取图片边界对应的下标max_w
    image = image[:, :max_h, :max_w]
    return image


def letterbox_image(image, image_size):
    """
    图片等比例缩放
    :param image: PIL image
    :param image_shape: (W, H)
    :return:
    """
    # 获取原始宽高和需要的宽高
    old_width, old_height = image.size
    new_width, new_height = image_size
    # 缩放图片有效区域
    scale = min(new_width / old_width, new_height / old_height)  # 图片有效区域缩放比例
    valid_width, valid_height = int(old_width * scale), int(old_height * scale)
    image = image.resize((valid_width, valid_height))
    # 填充图片无效区域
    origin = [(new_width - valid_width) // 2, (new_height - valid_height) // 2]
    result = Image.new(mode=image.mode, size=(new_width, new_height), color=(128, 128, 128))
    result.paste(image, origin)
    return result


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model weights path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    x, y = get_test_data(opt)
    # 模型
    num_classes = len(classes)
    model = get_model_sematic_segmentation(num_classes).to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)

        # 处理结果数据
        normalized_masks = pred['out'].softmax(dim=1).to('cpu')
        num_pics, num_classes, _, _ = normalized_masks.shape
        masks = (normalized_masks.argmax(1) == torch.arange(normalized_masks.shape[1])[:, None, None, None]).swapaxes(0, 1)  # 生成bool值
        labels = classes  # 不去除背景
        # masks = masks[:, 1:, :, :]  # 去除背景
        # labels = classes[1:]  # 去除背景

        # 标签数字转化为标签名称
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
        color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
        colors = color  # 不去除背景
        # colors = color[1:]  # 去除背景
        show_sematic_segmentation_result(x, masks, labels, image_size=[640, 640], colors=colors)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
