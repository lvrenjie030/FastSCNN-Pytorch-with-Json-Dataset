import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter


class NotSeawaterSegmentation(data.Dataset):
    NUM_CLASS = 2  # 我的数据集有两类

    def __init__(self, root='E:\yolov5-5.0\Maritime_ship_detection\Fast_SCNN\\No_Seawater_Dataset', split='train',
                 mode=None, transform=None,
                 base_size=640, crop_size=480, **kwargs):
        super(NotSeawaterSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("在文件夹中发现了0个文件，其地址：" + self.root + "\n")

            # 暂时不清楚有什么用

        self.valid_classes = [0, 1]  # 可用的类别，其中0代表背景，1代表目标
        self._key = np.array([1, 0], dtype='int32')  # 将掩码中的像素值转换为类别ID
        self._mapping = np.array([0, 255], dtype='int32')

    def __getitem__(self, index):
        # 此部分，读出了图片和掩码
        img = Image.open(self.images[index]).convert('RGB')  # 取出图片并转换为RGB格式
        if self.mode == 'test':  # 测试模式
            if self.transform is not None:
                img = self.transform(img)  # 调用train里面的转换函数
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])  # 取出掩码

        # 此部分，对图片和掩码进行同步变换
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # 以50%的概率对图片进行左右翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)  # 同步翻转

        # 此处对图像进行从0.5-2.0倍之间的随机缩放
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 从图像高宽的50%-200%之间随机选择一个整数
        w, h = img.size  # 获取图片的宽高
        # 根据图片的宽高比例，计算出缩放后的宽高
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)  # 双线性差值的放缩法
        mask = mask.resize((ow, oh), Image.NEAREST)  # 最近邻插值的放缩法

        # 填充后裁剪
        crop_size = self.crop_size
        if short_size < crop_size:
            # 假如缩放后的图片尺寸太小了，则进行填充
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # 随机的裁剪图片
        w, h = img.size  # 获取的是缩放+(可能填充)后的图片尺寸
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # 以50%的概率对图片进行高斯模糊
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)  # img变成numpy数组，mask变成
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')  # 先将mask转换为int32的数组，然后调用_class_to_index函数把mask中的像素值转换为类别ID
        target = target / 255  # 把掩码中的像素值转换为0和1（0,255--->0,1）
        target = target.astype('int32')
        return torch.LongTensor(target)  # 此处结果的维度为1

    def __len__(self):
        return len(self.images)

    # 用@property装饰器将方法变成属性
    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_pairs(folder, split='train'):
    def get_path_pairs(folder):  # 获取图像和掩码的路径
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    img_paths.append(img_path)
                elif file.endswith('.png'):
                    mask_path = os.path.join(root, file)
                    mask_paths.append(mask_path)
        return img_paths, mask_paths

    # 根据split参数选择文件夹
    if split == 'train':
        folder = os.path.join(folder, 'train')
    elif split == 'val':
        folder = os.path.join(folder, 'valid')
    elif split == 'test':
        folder = os.path.join(folder, 'test')
    elif split == 'trainval':
        print('trainval set')
        folder_train = os.path.join(folder, 'train')
        folder_val = os.path.join(folder, 'valid')
        img_paths_train, mask_paths_train = get_path_pairs(folder_train)
        img_paths_val, mask_paths_val = get_path_pairs(folder_val)
        return img_paths_train + img_paths_val, mask_paths_train + mask_paths_val
    else:
        raise ValueError(f"Unknown split: {split}")

    # 获取图像和掩码路径
    img_paths, mask_paths = get_path_pairs(folder)
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = NotSeawaterSegmentation()
    img, label = dataset[0]
    print(label.sum())
