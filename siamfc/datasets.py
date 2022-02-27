from __future__ import absolute_import, division

import numpy as np
import cv2
from glob import glob
import os
import pickle

from torch.utils.data import Dataset


class Got10kCropped(Dataset):

    def __init__(self, dataset_path, transforms=None,
                 pair_per_seq=1):
        super(Got10kCropped, self).__init__()
        self.transforms = transforms
        self.pairs_per_seq = pair_per_seq

        # 读取数据集所包含的视频序列，元数据，噪声标签，目标在搜索图像中的长宽比例
        with open(os.path.join(dataset_path, 'list.txt')) as f:
            seqs = f.readlines()
        seqs = [os.path.join(dataset_path, x.replace('\n','')) for x in seqs]
        self.seqs = seqs
        # 加载视频序列的元数据
        # meta_data = []
        # meta_data_names = [os.path.join(x, 'meta_data.txt') for x in self.seqs]
        # for meta_data_name in meta_data_names:
        #     with open(meta_data_name, 'rb') as f:
        #         meta_data.append( pickle.load(f) )
        # self.meta_data = meta_data
        # # 加载视频序列的标签
        # noisy_label = []
        # noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in self.seqs]
        # for noisy_label_name in noisy_label_names:
        #     with open(noisy_label_name, 'rb') as f:
        #         noisy_label.append(pickle.load(f))
        # self.noisy_label = noisy_label
        #
        # # 加载目标在搜索图像中的长宽比例
        # target_wh = []
        # target_wh_names = [os.path.join(x, 'target_wh.txt') for x in self.seqs]
        # for target_wh_name in target_wh_names:
        #     with open(target_wh_name, 'rb') as f:
        #         target_wh.append(pickle.load(f))
        # self.target_wh = target_wh

        print('loading metadata from:'+os.path.join(dataset_path, 'got10k_meta.pckl')+'\n')
        with open(os.path.join(dataset_path, 'got10k_meta.pckl'), 'rb') as f:
            got10k_meta = pickle.load(f)
        self.meta_data = got10k_meta['meta_data']
        self.noisy_label = got10k_meta['noisy_label']
        self.target_wh = got10k_meta['target_wh']

        self.indices = np.random.permutation(len(self.seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)] # 获得传入视频索引
        img_files = glob(os.path.join(self.seqs[index], '*.jpg'))
        noisy_label = self.noisy_label[index]
        meta = self.meta_data[index]
        target_wh = self.target_wh[index]

        # 获得滤除噪声序列后的视频序列标签。
        # with open(noisy_label, 'rb') as f:
        #     noisy_label = pickle.load(f)
        val_indices = np.logical_and.reduce(noisy_label)
        val_indices = np.where(val_indices)[0]

        # 如果当前视频序列中，满足筛选条件的视频帧小于2，则迭代寻找合适的视频。
        if len(val_indices)<2:
            index = np.random.choice(len(self.indices))
            return self.__getitem__(index)

        # 从视频序列中随机选取样本，并返回读取和变换后的图像序列。
        rand_z, rand_x = self._sample_pair((val_indices))

        # 读取视频帧，并根据要求的变换对视频帧进行变换
        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        item = (z, x)
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.seqs) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):        # 进行最大100次的循环采样，如果采样得到的训练对之间的间隔低于100，则选用此训练对，如果超过采样次数仍然没有获得满足条件的样本对，则随机算则一个样本对。
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x


class ILSVRC2015Cropped(Dataset):

    def __init__(self, dataset_path, transforms=None,
                 pair_per_seq=1):
        super(ILSVRC2015Cropped, self).__init__()
        self.transforms = transforms
        self.pairs_per_seq = pair_per_seq

        # 读取数据集所包含的视频序列，元数据，噪声标签，目标在搜索图像中的长宽比例
        with open(os.path.join(dataset_path, 'list.txt')) as f:
            seqs = f.readlines()
        seqs = [os.path.join(dataset_path, x.replace('\n','')) for x in seqs]
        self.seqs = seqs
        # 加载视频序列的元数据
        meta_data = []
        meta_data_names = [os.path.join(x, 'meta_data.txt') for x in self.seqs]
        for meta_data_name in meta_data_names:
            with open(meta_data_name, 'rb') as f:
                meta_data.append( pickle.load(f) )
        self.meta_data = meta_data
        # 加载视频序列的标签
        noisy_label = []
        noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in self.seqs]
        for noisy_label_name in noisy_label_names:
            with open(noisy_label_name, 'rb') as f:
                noisy_label.append(pickle.load(f))
        self.noisy_label = noisy_label

        # 加载目标在搜索图像中的长宽比例
        target_wh = []
        target_wh_names = [os.path.join(x, 'target_wh.txt') for x in self.seqs]
        for target_wh_name in target_wh_names:
            with open(target_wh_name, 'rb') as f:
                target_wh.append(pickle.load(f))
        self.target_wh = target_wh

        self.indices = np.random.permutation(len(self.seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)] # 获得传入视频索引
        img_files = glob(os.path.join(self.seqs[index], '*.JPEG'))
        noisy_label = self.noisy_label[index]
        meta = self.meta_data[index]
        target_wh = self.target_wh[index]

        # 获得滤除噪声序列后的视频序列标签。
        # with open(noisy_label, 'rb') as f:
        #     noisy_label = pickle.load(f)
        val_indices = np.logical_and.reduce(noisy_label)
        val_indices = np.where(val_indices)[0]

        # 如果当前视频序列中，满足筛选条件的视频帧小于2，则迭代寻找合适的视频。
        if len(val_indices)<2:
            index = np.random.choice(len(self.indices))
            return self.__getitem__(index)

        # 从视频序列中随机选取样本，并返回读取和变换后的图像序列。
        rand_z, rand_x = self._sample_pair((val_indices))

        # 读取视频帧，并根据要求的变换对视频帧进行变换
        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        item = (z, x)
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.seqs) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):        # 进行最大100次的循环采样，如果采样得到的训练对之间的间隔低于100，则选用此训练对，如果超过采样次数仍然没有获得满足条件的样本对，则随机算则一个样本对。
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x


class LaSOTCropped(Dataset):          # 需要统计LaSOT的视频序列平均长度和另外两个视频序列的平均长度再做定夺。
    def __init__(self, dataset_path, transforms=None,
                 pair_per_seq=1):
        super(LaSOTCropped, self).__init__()
        self.transforms = transforms
        self.pairs_per_seq = pair_per_seq

        # 读取数据集所包含的视频序列，元数据，噪声标签，目标在搜索图像中的长宽比例
        with open(os.path.join(dataset_path, 'list.txt')) as f:
            seqs = f.readlines()
        seqs = [os.path.join(dataset_path, x.replace('\n','')) for x in seqs]
        self.seqs = seqs
        # 加载视频序列的元数据
        meta_data = []
        meta_data_names = [os.path.join(x, 'meta_data.txt') for x in self.seqs]
        for meta_data_name in meta_data_names:
            with open(meta_data_name, 'rb') as f:
                meta_data.append( pickle.load(f) )
        self.meta_data = meta_data
        # 加载视频序列的标签
        noisy_label = []
        noisy_label_names = [os.path.join(x, 'noisy_label.txt') for x in self.seqs]
        for noisy_label_name in noisy_label_names:
            with open(noisy_label_name, 'rb') as f:
                noisy_label.append(pickle.load(f))
        self.noisy_label = noisy_label

        # 加载目标在搜索图像中的长宽比例
        target_wh = []
        target_wh_names = [os.path.join(x, 'target_wh.txt') for x in self.seqs]
        for target_wh_name in target_wh_names:
            with open(target_wh_name, 'rb') as f:
                target_wh.append(pickle.load(f))
        self.target_wh = target_wh

        self.indices = np.random.permutation(len(self.seqs))

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)] # 获得传入视频索引
        img_files = []
        for x in range(1, len(self.target_wh[index])+1):
            img_name = '%8d' % x
            img_name = img_name.replace(' ', '0') +'.jpg'
            img_file = os.path.join(self.seqs[index], img_name)
            img_files.append(img_file)

        # img_files = [os.path.join(self.seqs[index], ('%8d' % x).replace(' ', '0'))+'.jpg'
        #              for x in range(1, len(self.target_wh[index])+1)]
        # img_files = glob(os.path.join(self.seqs[index], '*.jpg'))
        noisy_label = self.noisy_label[index]
        meta = self.meta_data[index]
        target_wh = self.target_wh[index]

        # 获得滤除噪声序列后的视频序列标签。
        # with open(noisy_label, 'rb') as f:
        #     noisy_label = pickle.load(f)
        val_indices = np.logical_and.reduce(noisy_label)
        val_indices = np.where(val_indices)[0]

        # 如果当前视频序列中，满足筛选条件的视频帧小于2，则迭代寻找合适的视频。
        if len(val_indices)<2:
            index = np.random.choice(len(self.indices))
            return self.__getitem__(index)

        # 从视频序列中随机选取样本，并返回读取和变换后的图像序列。
        rand_z, rand_x = self._sample_pair((val_indices))

        # 读取视频帧，并根据要求的变换对视频帧进行变换
        try:
            z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
            x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
            z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        except:
            print('sequence name:{} z:{} x:{}'.format(self.seqs[index], rand_z, rand_x))

        item = (z, x)
        if self.transforms is not None:
            item = self.transforms(*item)
        return item

    def __len__(self):
        return len(self.seqs) * self.pairs_per_seq

    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):        # 进行最大100次的循环采样，如果采样得到的训练对之间的间隔低于100，则选用此训练对，如果超过采样次数仍然没有获得满足条件的样本对，则随机算则一个样本对。
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x









