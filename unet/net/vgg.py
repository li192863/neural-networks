import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features  # nn.Sequential
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):  # 3x512x512
        # x = self.features(x)  # 512x16x16
        # x = self.avgpool(x)  # 512x7x7
        # x = torch.flatten(x, 1)  # (512*7*7)
        # x = self.classifier(x)  # num_classes
        feat1 = self.features[:4](x)  # 64x512x512
        feat2 = self.features[4:9](feat1)  # 128x256x256
        feat3 = self.features[9:16](feat2)  # 256x128x128
        feat4 = self.features[16:23](feat3)  # 512x64x64
        feat5 = self.features[23:-1](feat4)  # 512x32x32
        return [feat1, feat2, feat3, feat4, feat5]  # (64x512x512, 128x256x256, 256x128x128, 512x64x64, 512x32x32)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':  # 最大池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 特征图大小减半
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 特征图大小不发生改变
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs['D'], batch_norm=False, in_channels=in_channels, **kwargs))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./data")
        model.load_state_dict(state_dict)
    del model.avgpool
    del model.classifier
    return model

if __name__ == '__main__':
    model = VGG16(False)
    print(model)
