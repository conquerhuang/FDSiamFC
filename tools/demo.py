from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerFDSiamFC


if __name__ == '__main__':
    # demo video, you can copy any video sequences from OTB dataset and put it into videos folder as the demo video.
    seq_dir = os.path.expanduser(r'../videos/Basketball/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')

    # #  demo for FDSiamFC_15
    # # set the Sq parameter under each convolutional layer in SiamFC's backbone network.
    # squeeze_rate = [0.15,  0.15, 0.15, 0.15, 0.15]
    #
    # # loading pretrained model from local disk and run FDSiamFC_15. you shall download it from netdisk.
    # net_path = r'./transformed_model_old_torch.pth'
    # tracker = TrackerFDSiamFC(net_path=net_path, squeeze_rate=squeeze_rate)
    # # run demo video.
    # tracker.track(img_files, anno[0], visualize=True)

    # demo for FDSiamFC_30
    # set the Sq parameter under each convolutional layer in SiamFC's backbone network.
    squeeze_rate = [0.3,  0.3, 0.3, 0.3, 0.3]

    # loading pretrained model from local disk and run FDSiamFC_30. you shall download it from netdisk.
    net_path = r'./transformed_model_30_old_torch.pth'
    tracker = TrackerFDSiamFC(net_path=net_path, squeeze_rate=squeeze_rate)
    # run demo video.
    tracker.track(img_files, anno[0], visualize=True)

