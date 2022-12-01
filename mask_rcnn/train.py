import argparse

import torch

import utils as utils
import transforms as T
from dataset import PennFudanDataset
from engine import train_one_epoch, evaluate
from mask_rcnn.model import get_model_instance_segmentation

NUM_CLASSES = 2  # background(0) and person(1)
DATASET_ROOT_PATH = '../../datasets/PennFudanPed/'
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'


def get_transform(train):
    """
    获取数据变换器
    :param train:
    :return:
    """
    transforms = []
    # 将图像转换为张量
    transforms.append(T.ToTensor())
    if train:
        # 训练过程中，随机翻转训练图片以及其真实值
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    # 使用数据集
    train_data = PennFudanDataset(DATASET_ROOT_PATH, get_transform(train=True))
    test_data = PennFudanDataset(DATASET_ROOT_PATH, get_transform(train=False))

    # 划分数据集为训练集与测试集
    indices = torch.randperm(len(train_data)).tolist()
    train_data = torch.utils.data.Subset(train_data, indices[:-50])
    test_data = torch.utils.data.Subset(test_data, indices[-50:])

    # 定义数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                                   collate_fn=utils.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                                  num_workers=4, collate_fn=utils.collate_fn)
    return train_dataloader, test_dataloader


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
    train_dataloader, test_dataloader = get_dataloader(opt)
    # 模型
    num_classes = 2
    model = get_model_instance_segmentation(num_classes).to(opt.device)
    # 参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.005 / opt.epochs)  # 优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # 训练
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train_one_epoch(model, optimizer, train_dataloader, opt.device, opt.epoch, print_freq=10)  # 训练
        lr_scheduler.step()  # 更新学习率
        evaluate(model, test_dataloader, opt.device)  # 测试
    print(f'Done!')
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    # # 测试模型
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(data='DEFAULT')
    # dataset = PennFudanDataset('../../datasets/PennFudanPed/', get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn
    # )
    # # 训练
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images, targets)  # Returns losses and detections
    # print(output)
    # # 推理
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions)
