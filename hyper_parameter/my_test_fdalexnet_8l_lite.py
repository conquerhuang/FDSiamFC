import os
from got10k.experiments import ExperimentOTB, ExperimentDTB70, ExperimentTColor128, ExperimentUAV123, ExperimentLaSOT, ExperimentGOT10k
from siamfc import TrackerFDSiamFC_8l_lite

from multiprocessing import Pool


def evaluate(model_path, tracker_name, experiment, squeeze_rate, scale_step,
             scale_lr, scale_penalty, window_influence, scale_num):
    tracker = TrackerFDSiamFC_8l_lite(
        net_path=model_path, name=tracker_name, squeeze_rate=squeeze_rate,
        scale_step = scale_step,
        scale_lr = scale_lr,
        scale_penalty = scale_penalty,
        window_influence = window_influence,
        scale_num=scale_num
    )
    # for experiment in experiments:
    experiment.run(tracker,)
    experiment.report([tracker_name])


if __name__ == '__main__':
    experiments = [
        # ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2015),
        # ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2013),
        # ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='tb50'),
        # ExperimentDTB70(root_dir=r'E:\dataset\tracker_evaluate_dataset\DTB70'),
        # ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128'),
        # ExperimentGOT10k(root_dir=r'E:\dataset\train_dataset\GOT-10K\full_data', subset='test'),
        # ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123'),
        # ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV20L')
        ExperimentLaSOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\LaSOT', subset='extension_train')
    ]

    # # 单线程
    # for i in range(1, 51):
    #     model_path = r'../models/siamfc_' + str(i) + '.pth'
    #     tracker_name = 'SiamFC_' + str(i)
    #     tracker = SiamFCTracker(model_path=model_path, tracker_name=tracker_name) #初始化一个追踪器
    #     for experiment in experiments:
    #         experiment.run(tracker)
    #         experiment.report(tracker_name)

    # # 多线程 多模型
    # tracker_infos = []
    # squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # for i in range(2, 52):
    #     model_path = r'../pretrained/transformed_model/FDSiam_8l_'+str(i)+'.pth'
    #     tracker_name = 'FDSiam_8l_' + str(i)
    #     for experiment in experiments:
    #         tracker_infos.append([model_path, tracker_name, experiment, squeeze_rate])
    # p = Pool(processes=4)
    # for tracker_info in tracker_infos:
    #     p.apply_async(evaluate, args=tuple(tracker_info))
    # p.close()
    # p.join()

    # # 多线程 单模型
    # tracker_infos = []
    #
    # model_path = r'../pretrained/SiamFC/transformed_model_double.pth'
    # tracker_name = 'SiamFC_8l_lite'
    # squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # for experiment in experiments:
    #     tracker_infos.append([model_path, tracker_name, experiment, squeeze_rate])
    # p = Pool(processes=4)
    # for tracker_info in tracker_infos:
    #     p.apply_async(evaluate, args=tuple(tracker_info))
    # p.close()
    # p.join()

    # # 单模型 单线程
    model_path = r'../pretrained/transformed_model/Finetune_49.pth'
    tracker_name = 'FDSiamFC_8l'
    squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    scale_num = 3
    scale_step = 1.087       # default 1.0375
    scale_lr = 0.26         # default 0.59
    scale_penalty = 0.978   # default 0.9745
    window_influence = 0.25       # default 0.176

    for experiment in experiments:
        evaluate(
            model_path, tracker_name, experiment, squeeze_rate,
            scale_step, scale_lr, scale_penalty, window_influence,
            scale_num
        )

    # tracker_name = 'SiamFC_8l_lite'
    # squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # for experiment in experiments:
    #     evaluate(model_path, tracker_name, experiment, squeeze_rate)

    # 多线程

