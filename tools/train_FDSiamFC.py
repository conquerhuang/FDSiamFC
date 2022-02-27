# from __future__ import absolute_import

import os
import numpy as np
import scipy.io as scio
import shutil
# from got10k.datasets import *

from siamfc.siamfc import TrackerSiamFC
from siamfc.datasets import Got10kCropped
from tools.fdModel_evaluate import modules_evaluate, best_result_otb2013


if __name__ == '__main__':

    root_dir = os.path.expanduser(r'E:\Cropped_GOT10K')
    train_dataset = Got10kCropped(root_dir, pair_per_seq=1)

    squeeze_rates = [[
        0.15,   # conv1
        0.15,   # conv2
        0.15,   # conv3
        0.15,   # conv4
        0.15    # conv5
        ]]
    for squeeze_iter, squeeze_rate in enumerate(squeeze_rates):
        total_turns = 1
        net_path = None
        for step in range(1, 10):
            for turn in range(total_turns):
                name = 'FDSiamFC_'+str(squeeze_iter)+'_'+str(step)+'_'+str(turn)
                if step == 0:
                    net_path = None
                    name = 'Finetune_' + str(squeeze_iter) + '_' + str(step) + '_' + str(turn)
                    tracker = TrackerSiamFC(net_path=net_path, name=name, squeeze_rate=squeeze_rate, step=step,
                                            train=True)
                    tracker.train_over(train_dataset)
                else:
                    net_path, AUC, DP, SR = best_result_otb2013(r'./reports', name_key_word='Finetune_' + str(
                        squeeze_iter) + '_' + str(step - 1))
                    # 训练跟踪器
                    tracker = TrackerSiamFC(net_path=net_path, name=name, squeeze_rate=squeeze_rate, step=step, train=True)
                    tracker.train_over(train_dataset)
                    # 评估模型，并得到最好的模型用于微调。
                    modules_evaluate(squeeze_iter=squeeze_iter, name_head='FDSiamFC',
                                     squeeze_rate=squeeze_rate, step=step,total_turns=total_turns)
                    net_path, AUC, DP, SR = best_result_otb2013(r'./reports',name_key_word='FDSiamFC_' + str(squeeze_iter) + '_' + str(step))
                    # 打印当前训练结果。
                    with open('train_log.txt', 'a+') as f:
                        f.write('train phase:')
                        f.write('squeeze iter :' + str(squeeze_iter) + '\n')
                        f.write('step :   ' + str(step) + '\n')
                        f.write('best model :  ' + net_path + '\n')
                        f.write('AUC:' + str(AUC) + '   DP:' + str(DP) + '   SR:' + str(SR) + '\n\n')

                    # 微调模型
                    print('fine tune model from previous trained model:')
                    fine_tune_model_path = os.path.join(r'../pretrained/temp', name+'_finetune.pth')
                    print(net_path)
                    shutil.copy(net_path, fine_tune_model_path)
                    # 微调, 微调时需要修改学习率，防止模型跳出之前预训练好的模型。 train=False是为了加载整个模型。
                    name = 'Finetune_' + str(squeeze_iter) + '_' + str(step) + '_' + str(turn)
                    tracker = TrackerSiamFC(net_path=fine_tune_model_path, name=name, squeeze_rate=squeeze_rate,
                                            step=step, train=False, fine_tune=True, initial_lr=1e-3)
                    tracker.train_over(train_dataset)
            # 评估跟踪器，并找到最好的模型，作为下一次训练的初始模型。
            modules_evaluate(squeeze_iter=squeeze_iter, name_head='Finetune',
                             squeeze_rate=squeeze_rate, step=step, total_turns=total_turns)
            net_path, AUC, DP, SR = best_result_otb2013(r'./reports', name_key_word='Finetune_'+str(squeeze_iter)+'_'+str(step))

            # 记录每次训练的结果
            with open('train_log.txt', 'a+') as f:
                f.write('fine tune phase:')
                f.write('squeeze iter :' + str(squeeze_iter) + '\n')
                f.write('step :   ' + str(step) + '\n')
                f.write('best model :  ' + net_path + '\n')
                f.write('AUC:' + str(AUC) + '   DP:' + str(DP) + '   SR:' + str(SR) + '\n\n')







