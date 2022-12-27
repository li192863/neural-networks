import numpy as np
from torchvision.datasets import VOCSegmentation

from deeplabv3.presets import SegmentationPresetTrain

DATASET_ROOT_PATH = '../../datasets'
train_data = VOCSegmentation(DATASET_ROOT_PATH, '2012', 'train', download=False)
train_data1 = VOCSegmentation(DATASET_ROOT_PATH, '2012', 'train', download=False,
                                 transforms=SegmentationPresetTrain(base_size=520, crop_size=480))

x, y = train_data[1]
x.show()
y.show()

y_arr = np.asarray(y)

print('wait for it')