import numpy as np
import cv2
from glob import glob
import os
import pickle

from torch.utils.data import Dataset

def genIlsvrc15MetaData(dataset_path='', meta_name=''):
    # 读取数据集所包含的视频序列，元数据，噪声标签，目标在搜索图像中的长宽比例
    with open(os.path.join(dataset_path, 'list.txt')) as f:
        seqs = f.readlines()
    seqs = [os.path.join(dataset_path, x.replace('\n', '')) for x in seqs]
    # 加载视频序列的元数据
    meta_data = []
    meta_data_names = [os.path.join(x, 'meta_data.txt') for x in seqs]
    for meta_data_name in meta_data_names:
        with open(meta_data_name, 'rb') as f:
            meta_data.append(pickle.load(f))
    # 加载视频序列的标签
    noisy_label = []
    noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in seqs]
    for noisy_label_name in noisy_label_names:
        with open(noisy_label_name, 'rb') as f:
            noisy_label.append(pickle.load(f))

    # 加载目标在搜索图像中的长宽比例
    target_wh = []
    target_wh_names = [os.path.join(x, 'target_wh.txt') for x in seqs]
    for target_wh_name in target_wh_names:
        with open(target_wh_name, 'rb') as f:
            target_wh.append(pickle.load(f))

    meta_data_save = {}
    meta_data_save['meta_data'] = meta_data
    meta_data_save['noisy_label'] = noisy_label
    meta_data_save['target_wh'] = target_wh

    with open(os.path.join(dataset_path, meta_name), 'wb') as f:
        pickle.dump(meta_data_save, f)


def TrackingNetMetaData(dataset_path='', meta_name=''):
    # 读取数据集所包含的视频序列，元数据，噪声标签，目标在搜索图像中的长宽比例
    with open(os.path.join(dataset_path, 'list.txt')) as f:
        seqs = f.readlines()
    seqs = [os.path.join(dataset_path, x.replace('\n', '')) for x in seqs]
    # 加载视频序列的元数据
    meta_data = []
    meta_data_names = [os.path.join(x, 'meta_data.txt') for x in seqs]
    for meta_data_name in meta_data_names:
        with open(meta_data_name, 'rb') as f:
            meta_data.append(pickle.load(f))
    # 加载视频序列的标签
    noisy_label = []
    noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in seqs]
    for noisy_label_name in noisy_label_names:
        with open(noisy_label_name, 'rb') as f:
            noisy_label.append(pickle.load(f))

    # 加载目标在搜索图像中的长宽比例
    target_wh = []
    target_wh_names = [os.path.join(x, 'target_wh.txt') for x in seqs]
    for target_wh_name in target_wh_names:
        with open(target_wh_name, 'rb') as f:
            target_wh.append(pickle.load(f))

    meta_data_save = {}
    meta_data_save['meta_data'] = meta_data
    meta_data_save['noisy_label'] = noisy_label
    meta_data_save['target_wh'] = target_wh

    with open(os.path.join(dataset_path, meta_name), 'wb') as f:
        pickle.dump(meta_data_save, f)



if __name__ == '__main__':
    # dataset_path = r'E:\Cropped_ILSVRC2015'
    # meta_name = 'ilsvrc15_meta.pickl'
    # genIlsvrc15MetaData(dataset_path, meta_name)

    dataset_path = r'E:\Cropped_TrackingNet'
    meta_name = 'TrackingNet_meta.pickl'
    genIlsvrc15MetaData(dataset_path, meta_name)









