import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # 加载在COCO上预训练的实例分割模型mask-rcnn
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT', weights_backbone='DEFAULT')

    # 获取分类器的输入种类数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 将预测器头替换为新的
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取蒙版分类器的输入种类数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 将蒙版的预测器头替换为新的
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

if __name__ == '__main__':
    model = get_model_instance_segmentation(2)  # 1 class (person) + background

    print(model)