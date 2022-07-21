from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerFDSiamFC

if __name__ == '__main__':
    seq_dir = os.path.expanduser(r'G:/dataset/tracker_evaluate_dataset/OTB/Basketball/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')

    # # tracker with squeeze rate = 0.3
    # model_path = r'../pretrained/FDSiamfc_30.pth'
    # squeeze_rate = [0.3, 0.3, 0.3, 0.3, 0.3]
    # tracker = TrackerFDSiamFC(net_path=model_path, squeeze_rate=squeeze_rate)
    # tracker.track(img_files, anno[0], visualize=True)

    # tracker with squeeze rate = 0.15
    model_path = r'../pretrained/FDSiamfc_15.pth'
    squeeze_rate = [0.15, 0.15, 0.15, 0.15, 0.15]
    tracker = TrackerFDSiamFC(net_path=model_path, squeeze_rate=squeeze_rate)
    tracker.track(img_files, anno[0], visualize=True)