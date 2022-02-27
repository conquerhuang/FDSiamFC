import os
from got10k.experiments import *
from siamfc import TrackerFDSiamFC

from multiprocessing import Pool

def evaluate(model_path, tracker_name, experiment, squeeze_rate):
    tracker = TrackerFDSiamFC(net_path=model_path, name=tracker_name, squeeze_rate=squeeze_rate)
    # for experiment in experiments:
    experiment.run(tracker)
    experiment.report([tracker_name])


if __name__ == '__main__':
    experiments = [
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2015),
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2013),
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='tb50'),
        ExperimentDTB70(root_dir=r'E:\dataset\tracker_evaluate_dataset\DTB70'),
        ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128'),
        # ExperimentGOT10k(root_dir=r'E:\dataset\train_dataset\GOT-10K\full_data'),
        ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123'),
        ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV20L')
        # ExperimentLaSOT(root_dir='')
    ]

    # # 单线程
    # for i in range(1, 51):
    #     model_path = r'../models/siamfc_' + str(i) + '.pth'
    #     tracker_name = 'SiamFC_' + str(i)
    #     tracker = SiamFCTracker(model_path=model_path, tracker_name=tracker_name) #初始化一个追踪器
    #     for experiment in experiments:
    #         experiment.run(tracker)
    #         experiment.report(tracker_name)

    # # 多线程
    # tracker_infos = []
    # for i in range(21, 51):
    #     model_path = r'../pretrained/SiamFC_new/siamfc_alexnet_e'+ str(i) + '.pth'
    #     tracker_name = 'SiamFC_' + str(i)
    #     for experiment in experiments:
    #         tracker_infos.append([model_path, tracker_name, experiment])
    # p = Pool(processes=3)
    # for tracker_info in tracker_infos:
    #     p.apply_async(evaluate, args=tuple(tracker_info))
    # p.close()
    # p.join()

    # # 多线程, 单跟踪器
    # tracker_infos = []
    #
    # squeeze_rate = 0.2
    # model_path = r'..\pretrained\SiamFC\transformed_model_20.pth'
    # tracker_name = 'FDSiamFC_20'
    # for experiment in experiments:
    #     tracker_infos.append([model_path, tracker_name, experiment, squeeze_rate])
    # p = Pool(processes=5)
    # for tracker_info in tracker_infos:
    #     p.apply_async(evaluate, args=tuple(tracker_info))
    # p.close()
    # p.join()

    # 单线程, 单跟踪器
    tracker_infos = []

    squeeze_rate = [0.15,  0.15, 0.15, 0.15, 0.15]
    model_path = r'../pretrained/transformed_model_15_old_torch.pth'
    tracker_name = 'FDSiamFC_15'
    for experiment in experiments:
        evaluate(model_path, tracker_name, experiment, squeeze_rate)
