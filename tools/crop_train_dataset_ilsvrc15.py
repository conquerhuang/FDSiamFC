from __future__ import absolute_import

import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from siamfc import ops
from got10k.datasets import *

instance_size = 255
context = 0.5
exemplar_sz = 127

ILSVRC2015_dir = r'G:\dataset\train_dataset\ILSVRC2015'
cropped_data_dir = r'G:\dataset\train_dataset\Cropped_ILSVRC2015'
seqs = ImageNetVID(ILSVRC2015_dir, subset=('train', 'val'))


def dataset_filter(img0, anno, vis_ratios=None):
    # 滤除训练集中的噪声数据
    size = np.array(img0.shape[1::-1])[np.newaxis, :]
    areas = anno[:, 2] * anno[:, 3]

    # acceptance conditions
    c1 = areas >= 20  # 目标的像素面积大于20
    c2 = np.all(anno[:, 2:] >= 20, axis=1)  # np.all 判断在维度axis（axis=1,表示的是行）上所有的元素是否都为True，此处判断长宽是否都超过20
    c3 = np.all(anno[:, 2:] <= 500, axis=1)  # 目标的长宽像素是否均小于500
    c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)  # 目标的长宽与整个图片的比值是否大于0.01
    c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)  # 目标的长宽与整个图片的比值是否小于0.5
    c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25  # 宽度和高度的比值是否大于0.25
    c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4  # 宽度和高度的比值是否小于4
    if vis_ratios is not None:
        c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))  # 目标的可见区域是否超过最大可见区域大小的0.3倍。
    else:
        c8 = np.ones_like(c1)

    mask = np.logical_and.reduce(  # np.logical_and numpy 的逻辑与运算，reduce用于实现多列逻辑按位与运算。
        (c1, c2, c3, c4, c5, c6, c7, c8))
    # val_indices = np.where(mask)[0]  # 得到训练视频下，满足c1到c7所定义的7个条件的帧。
    return (c1, c2, c3, c4, c5, c6, c7, c8)


def crop_image(img, box, out_size, context=0.5, exemplar_sz=127):
    # convert box to 0-indexed and center based [y, x, h, w]
    box = np.array([  # 转换边界框排序方式为（中心x,中心y,高度h,宽度w）
        box[1] - 1 + (box[3] - 1) / 2,
        box[0] - 1 + (box[2] - 1) / 2,
        box[3], box[2]], dtype=np.float32)
    center, target_sz = box[:2], box[2:]  # center 目标中心， target_sz 目标尺寸(高度h，宽度w)

    context = context * np.sum(target_sz)  # 上下文(周围环绕区域)所对应的大小。
    size = np.sqrt(np.prod(target_sz + context))  # 计算目标包含上下文时的大小(像素面积)
    size *= out_size / exemplar_sz  # 计算搜索图像对应的像素面积。

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


def crop_dataset(meta_exist, seq_index, cropped_data_dir):
    #  裁剪视频序列。
    # 获取视频序列的信息。
    print('processing:{} / {}'.format(seq_index, len(seqs)))
    if meta_exist:
        image_files, anno, meta = seqs[seq_index]
        vis_ratios = meta.get('cover', None)
    else:
        image_files, anno = seqs[seq_index][:2]
        vis_ratios = None
    # seq_name = image_files[0].split('\\')[-2]
    seq_name = seqs.seq_names[seq_index]
    if not os.path.exists(os.path.join(cropped_data_dir, seq_name)):
        os.makedirs(os.path.join(cropped_data_dir, seq_name))

    # 获得噪声帧信息标签。  需要修改
    noisy_label = dataset_filter(
        cv2.imread(image_files[0], cv2.IMREAD_COLOR),
        anno,
        vis_ratios
    )

    # 存视频序列下的   可接受条件（噪声标签）  该视频序列所对应的元数据   目标在裁剪图像中的宽高
    with open(os.path.join(cropped_data_dir, seq_name, 'noisy_label.txt'), 'wb') as f:
        pickle.dump(noisy_label, f)  # 存储噪声标签
    with open(os.path.join(cropped_data_dir, seq_name, 'meta_data.txt'), 'wb') as f:
        meta = {'resolution': np.array(cv2.imread(image_files[0], cv2.IMREAD_COLOR).shape[1::-1])[np.newaxis, :]}
        pickle.dump(meta, f)  # 存储元数据
    # 计算目标对象在裁剪，缩放，填充后的训练样本中的宽度和高度。
    size_w = anno[:, 2] + (anno[:, 2] + anno[:, 3]) / 2
    size_h = anno[:, 3] + (anno[:, 2] + anno[:, 3]) / 2
    target_ratio = exemplar_sz / np.sqrt(size_w * size_h)
    target_wh = anno[:, 2:] * np.stack((target_ratio, target_ratio), axis=1)
    with open(os.path.join(cropped_data_dir, seq_name, 'target_wh.txt'), 'wb') as f:
        pickle.dump(target_wh, f)

    # 对当前视频序列进行裁剪,并储存裁剪后的信息。
    for image_file_index in range(len(image_files)):
        image_name = image_files[image_file_index].split('\\')[-1]
        image_file = cv2.imread(image_files[image_file_index], cv2.IMREAD_COLOR)
        image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
        box = anno[image_file_index]

        # 裁剪用于后续训练的目标图像， 一共缩放了 exemplar_size / ( sqrt(w+(w+h)/2) (h+(w+h)/2) )
        cropped_image = crop_image(image_file, box, instance_size, context, exemplar_sz)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(cropped_data_dir, seq_name, image_name), cropped_image)

if __name__ == '__main__':

    meta_exist = getattr(seqs, 'return_meta', False)

    # 使用多线程对视频序列进行裁剪
    seq_infos = []
    for seq_index in range(len(seqs)):
        seq_infos.append([meta_exist, seq_index, cropped_data_dir])
    p = Pool(processes=14)
    for seq_info in seq_infos:
        p.apply_async(crop_dataset, args=tuple(seq_info))
    p.close()
    p.join()

    # # 使用单线程对样本进行裁剪
    # for seq_index in tqdm(range(len(seqs))):    # 对视频帧进行裁剪
    #     crop_dataset(meta_exist, seq_index, cropped_data_dir)



