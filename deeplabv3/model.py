import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


def get_model_sematic_segmentation(num_classes):
    # 加载在COCO上预训练的实例分割模型mask-rcnn
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT', aux_loss=True, weights_backbone='DEFAULT')

    # 获取分类器的输入种类数
    in_channels = model.classifier[0].convs[0][0].in_channels
    # 将预测器替换为新的
    model.classifier = DeepLabHead(in_channels, num_classes)

    # 获取辅助分类器的输入种类数
    in_channels_aux = model.aux_classifier[0].in_channels
    # 将辅助预测器替换为新的
    model.aux_classifier = FCNHead(in_channels_aux, num_classes)


    # # 获取分类器的输入种类数
    # in_channels = model.classifier[4].in_channels
    # # 将最后卷积层替换为新的
    # model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    #
    # # 获取辅助分类器的输入种类数
    # in_channels_aux = model.aux_classifier[4].in_channels
    # # 将最后卷积层替换为新的
    # model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model

if __name__ == '__main__':
    model = get_model_sematic_segmentation(6)

    print(model)