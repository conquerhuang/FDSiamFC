# from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
import torch

from siamfc import ops

# __all__ = ['SiamFCTransforms']


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, img):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale))
        return cv2.resize(img, out_size, interpolation=interp)


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad

        return img[i:i + th, j:j + tw]


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


class ToTensor(object):

    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0, 1))


class SiamFCTransforms(object):

    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        # 对模板进行变换，随机拉升(均匀分布)，中心裁剪，随机裁剪，中心裁剪(裁剪为样本尺寸)，转换为张量。
        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            CenterCrop(exemplar_sz),
            ToTensor()])
        # 对搜索图像进行变换， 随机拉升(均匀分布)，中心裁剪，随机裁剪，转换为张量。
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            ToTensor()])

    def __call__(self, z, x):  # 对输入样本进行随机拉升，随机裁剪，并转换为张量，用于后续训练。
        z = self.transforms_z(z)
        x = self.transforms_x(x)
        return z, x

    def _crop(self, img, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([  # 转换边界框排序方式为（中心x,中心y,高度h,宽度w）
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]  # center 目标中心， target_sz 目标尺寸(高度h，宽度w)

        context = self.context * np.sum(target_sz)  # 上下文(周围环绕区域)所对应的大小。
        size = np.sqrt(np.prod(target_sz + context))  # 计算目标包含上下文时的大小(像素面积)
        size *= out_size / self.exemplar_sz  # 计算搜索图像对应的像素面积。

        avg_color = np.mean(img, axis=(0, 1), dtype=float)  # 平均颜色，用于索引超出图像边界时，对图像进行纯色填充。
        interp = np.random.choice([  # 插值方式从五种插值方式中随机选择一种方式进行插值。
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)

        return patch
