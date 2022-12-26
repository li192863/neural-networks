import colorsys
import os
from random import randint

import torch
from PIL import Image


class PeopleClothingDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载所有图片文件，并对文件进行排序
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'jpeg_images', 'IMAGES'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'jpeg_masks', 'MASKS'))))

    def __getitem__(self, idx):
        # 加载图片以及蒙版
        img_path = os.path.join(self.root, 'jpeg_images', 'IMAGES', self.imgs[idx])
        mask_path = os.path.join(self.root, 'jpeg_masks', 'MASKS', self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    classes = ['null', 'accessories', 'bag', 'belt', 'blazer', 'blouse', 'bodysuit', 'boots', 'bra', 'bracelet', 'cape',
               'cardigan', 'clogs', 'coat', 'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair', 'hat', 'heels',
               'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings', 'loafers', 'necklace', 'panties', 'pants',
               'pumps', 'purse', 'ring', 'romper', 'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt',
               'sneakers', 'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear', 't-shirt',
               'tie', 'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
    num_classes = len(classes)
    print(len(classes))

    # image = Image.open('../../datasets/people-clothing-segmentation/jpeg_images/IMAGES/img_0002.jpeg')
    # image.show()
    # mask = Image.open('../../datasets/people-clothing-segmentation/jpeg_masks/MASKS/seg_0002.jpeg').convert('L')
    # 标签数字转化为标签名称
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
    color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
    colors = []
    for c in color:
        colors.append(c[0])
        colors.append(c[1])
        colors.append(c[2])

    root = '../../datasets/people-clothing-segmentation'
    dataset = PeopleClothingDataset(root)
    print(dataset[1])
    x, y = dataset[1]
    x.show()
    # y.convert('P')
    # y.putpalette(colors)
    y.show()

