import os
import numpy as np
import torch
import torch.utils.data

from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载所有图片文件，并对文件进行排序
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 加载图片以及蒙版
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # 注意并未将蒙版转为RGB，因为每个颜色对应一个不同的实例（0代表背景）

        mask = np.array(mask)
        obj_ids = np.unique(mask)  # 每个实例对应着不同的颜色，找出所有实例
        obj_ids = obj_ids[1:]  # 移除第一个，即背景实例

        masks = mask == obj_ids[:, None, None]  # 将蒙版转化为True/False的蒙版

        # 找出各个实例的边界框
        num_objs = len(obj_ids)  # 所有的实例个数
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])  # 返回True位置对应的下标组合
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 只有一个种类
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # 假定所有实例都不拥挤

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    image = Image.open('../../datasets/PennFudanPed/PNGImages/FudanPed00001.png')
    image.show()
    mask = Image.open('../../datasets/PennFudanPed/PedMasks/FudanPed00001_mask.png').convert('P')
    # 每个实例拥有不同的颜色，从0到N，N为实例数，为了方便显示，对mask填充颜色
    mask.putpalette([
        0, 0, 0,  # 0的位置填充黑色
        255, 0, 0,  # 1的位置填充红色
        255, 255, 0,  # 2的位置填充黄色
        255, 153, 0,  # 3的位置填充橘色
    ])
    mask.show()

    root = '../../datasets/PennFudanPed/'
    dataset = PennFudanDataset(root)
    print(dataset[0])
