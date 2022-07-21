import os
import numpy as np
from got10k.experiments import ExperimentOTB, ExperimentDTB70, ExperimentTColor128, ExperimentUAV123, ExperimentGOT10k
from siamfc import TrackerFDSiamFC_8l_lite
import json
import scipy.io as scio

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
        # ExperimentOTB(root_dir=r'G:\dataset\tracker_evaluate_dataset\OTB', version=2015),
        # ExperimentOTB(root_dir=r'G:\dataset\tracker_evaluate_dataset\OTB', version=2013),
        # ExperimentOTB(root_dir=r'G:\dataset\tracker_evaluate_dataset\OTB', version='tb50'),
        # ExperimentDTB70(root_dir=r'G:\dataset\tracker_evaluate_dataset\DTB70'),
        # ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128'),
        # ExperimentGOT10k(root_dir=r'E:\dataset\train_dataset\GOT-10K\full_data', subset='val'),
        # ExperimentUAV123(root_dir=r'G:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123'),
        ExperimentUAV123(root_dir=r'G:\dataset\tracker_evaluate_dataset\UAV123', version='UAV20L')
        # ExperimentLaSOT(root_dir=r'E:\dataset\tracker_evaluate_dataset\LaSOT', subset='test')
    ]

    # 多线程 单模型
    tracker_infos = []

    # base parameters
    model_path = r'../pretrained/FDSiamd.pth'
    squeeze_rate = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # finetune parameters

    scale_num = 3
    scale_step = 1.087       # default 1.0375
    scale_lr = 0.26         # default 0.59
    scale_penalty = 0.978   # default 0.9745
    window_influence = 0.176       # default 0.176

    # parameter for iteration
    window_influences = np.linspace(0.13, 0.28, 6)
    # 用于统计结果的变量
    hyper_parameters = np.zeros([len(window_influences), 9], dtype=np.double)
    tracker_names = []

    # start iteration and search the best hyper parameter.
    for i, window_influence in enumerate(window_influences): # tracker name should connect with hyper parameters.
        tracker_name = 'siamfc_8l_sst_' + str(int(scale_step*10000)) + \
                       '_slr_'+str(int(scale_lr*10000))+\
                       '_sp_' + str(int(scale_penalty*10000)) + \
                       '_winf_' + str(int(window_influence*10000))
        tracker_names.append(tracker_name)
        hyper_parameters[i, 0:5] = np.fromiter([scale_step, scale_lr, scale_penalty, window_influence, scale_num], dtype=np.double)

        # test on selected benchmarks
        for experiment in experiments:
            tracker_infos.append([
                model_path, tracker_name, experiment, squeeze_rate,
                scale_step, scale_lr, scale_penalty, window_influence,
                scale_num
            ])
    # for tracker_info in tracker_infos:
    #     evaluate(*tracker_info)

    p = Pool(processes=6)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()


    # collect results and save results
    tracker_results = []
    for tracker_name in tracker_names:
        tracker_dir = os.path.join(r'./reports', 'LaSOT_pt2', tracker_name, 'performance.json')
        with open(tracker_dir) as fp:
            result = json.load(fp)
            tracker_results.append(result)

    # other benchmark
    successScore = []
    precisionScore = []
    successRate = []

    for result in tracker_results:
        successScore.append(result[list(result.keys())[0]]['overall']['success_score'])
        precisionScore.append(result[list(result.keys())[0]]['overall']['precision_score'])
        successRate.append(result[list(result.keys())[0]]['overall']['success_rate'])
    scores = [0.4*a+0.4*b+0.2*c for a, b, c in zip(successScore, precisionScore, successRate)]
    hyper_parameters[:, 5] = np.fromiter(successScore, np.double)
    hyper_parameters[:, 6] = np.fromiter(precisionScore, np.double)
    hyper_parameters[:, 7] = np.fromiter(successRate, np.double)
    hyper_parameters[:, 8] = np.fromiter(scores, np.double)
    scio.savemat('results_LaSOT_pt2.mat', {'results': hyper_parameters})










