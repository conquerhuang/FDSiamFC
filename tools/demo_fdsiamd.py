from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerFDSiamFC_8l_lite

if __name__ == '__main__':
    seq_dir = os.path.expanduser(r'G:/dataset/tracker_evaluate_dataset/OTB/Basketball/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')

    model_path = r'../pretrained/FDSiamd.pth'
    squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    tracker = TrackerFDSiamFC_8l_lite(net_path=model_path, squeeze_rate=squeeze_rate)
    tracker.track(img_files, anno[0], visualize=True)
