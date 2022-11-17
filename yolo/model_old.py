import cv2
import torch
from torch import nn
from torch.autograd import Variable

from util import *


def parse_cfg(cfg_file):
    """
    解析cfg配置文件
    :param cfg_file:
    :return:
    """
    # 读取文件
    with open(cfg_file, 'r') as file:
        lines = file.read().split('\n')  # 存储文件行至列表中
        lines = [line for line in lines if len(line) > 0]  # 去除空行
        lines = [line for line in lines if not line.startswith('#')]  # 去除注释
        lines = [line.lstrip().rstrip() for line in lines]  # 去除左右空白符
    # 构造列表
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()  # '[net]'
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()  # 'batch = 64'
    blocks.append(block)  # 添加最后一个block
    return blocks


def create_modules(blocks):
    """
    创建模块
    :param blocks:
    :return:
    """
    net_info = blocks[0]  # 网络信息
    module_list = nn.ModuleList()  # 包含nn.Module的模块列表
    in_channels = 3  # 输入通路为3
    out_channels = []  # 记录输出通路个数
    filters = 0  # 初始化通道数
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            filters = int(block['filters'])  # 卷积层
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            if padding:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0
            try:  # 批标准化层
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            activation = block['activation']  # 激活层
            # 卷积层
            module.add_module(f'conv_{index}', nn.Conv2d(in_channels, filters, kernel_size, stride, padding, bias=bias))
            # 批标准化层
            if batch_normalize:
                module.add_module(f'batch_norm_{index}', nn.BatchNorm2d(filters))
            # 激活层（Linear或LeakyReLU）
            if activation == 'leaky':
                module.add_module(f'leaky_{index}', nn.LeakyReLU(0.1, inplace=True))
        elif block['type'] == 'upsample':
            stride = block['stride']
            # 上采样层
            module.add_module(f'upsample_{index}', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        elif block['type'] == 'route':
            layers = [int(layer) for layer in block['layers'].split(',')]
            if len(layers) == 1:
                start = layers[0]
                if start > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                    start = start - index
                filters = out_channels[index + start]
            else: # len(layers) == 2
                start, end = layers
                if start > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                    start = start - index
                if end > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                    end = end - index
                filters = out_channels[index + start] + out_channels[index + end]
            # 通路层
            module.add_module("route_{0}".format(index), EmptyLayer())
        elif block['type'] == 'shortcut':
            # 短连接层
            module.add_module(f'shortcut_{index}', EmptyLayer())
        elif block['type'] == 'yolo':
            mask = [int(x) for x in block['mask'].split(',')]
            anchors = [int(a) for a in block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            # 检测层
            module.add_module(f'Detection_{index}', DetectionLayer(anchors))

        module_list.append(module)
        in_channels = filters
        out_channels.append(filters)
    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, device):
        outputs = {}  # We cache the outputs for the route layer

        write = 0  # whether we have encountered the first detection or not
        for index, block in enumerate(self.blocks[1:]):
            block_type = block['type']

            if block_type == 'convolutional' or block_type == 'upsample':
                x = self.module_list[index](x)
            elif block_type == 'route':
                layers = [int(layer) for layer in block['layers'].split(',')]
                if len(layers) == 1:
                    start = layers[0]
                    if start > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                        start = start - index
                    x = outputs[index + start]
                else:  # len(layers) == 2
                    start, end = layers
                    if start > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                        start = start - index
                    if end > 0:  # 若为正数则转为倒数 倒数第i可视为-i
                        end = end - index
                    x = torch.cat((outputs[index + start], outputs[index + end]), dim=1)
            elif block_type == 'shortcut':
                x = outputs[index - 1] + outputs[index + int(block['from'])]
            elif block_type == 'yolo':
                anchors = self.module_list[index][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(block['classes'])

                x = x.to(device)
                x = predict_transform(x, inp_dim, anchors, num_classes, device)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), dim=1)

            outputs[index] = x

        return detections


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                 # Convert to Variable
    return img_

if __name__ == "__main__":
    # blocks = parse_cfg("cfg/yolov3.cfg")
    # print(blocks)
    # print(create_modules(blocks))

    model = Darknet("cfg/yolov3.cfg")
    inp = get_test_input()
    pred = model(inp, 'cuda')
    print(pred)
    print(model)
