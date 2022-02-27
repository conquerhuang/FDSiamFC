import os
import json
import numpy as np

from got10k.experiments import *
from siamfc import TrackerSiamFC
from multiprocessing import Pool


def evaluate(model_path, tracker_name, experiment, squeeze_rate=None, step=0):
    if squeeze_rate is None:
        squeeze_rate = [1, 1, 1, 1]
    tracker = TrackerSiamFC(net_path=model_path, name=tracker_name, squeeze_rate=squeeze_rate, step=step, train=False)
    # for experiment in experiments:
    experiment.run(tracker)
    experiment.report([tracker_name])

def modules_evaluate(squeeze_iter=1,squeeze_rate=None, step=0, total_turns=3, name_head='FDSiamFC'):
    experiments = [
        # ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2015),
        ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version=2013),
        # ExperimentOTB(root_dir=r'E:\dataset\tracker_evaluate_dataset\OTB', version='tb50'),
        # ExperimentDTB70(root_dir=r'E:\dataset\tracker_evaluate_dataset\DTB70'),
        # ExperimentTColor128(root_dir=r'E:\dataset\tracker_evaluate_dataset\Tcolor128'),
        # ExperimentGOT10k(root_dir=r'E:\dataset\train_dataset\GOT-10K\full_data'),
        # ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV123'),
        # ExperimentUAV123(root_dir=r'E:\dataset\tracker_evaluate_dataset\UAV123', version='UAV20L')
        # ExperimentLaSOT(root_dir='')
    ]
    if squeeze_rate is None:
        squeeze_rate = [1, 1, 1, 1]
    tracker_infos = []
    for turn in range(total_turns):
        for i in range(10, 51):
            model_path = r'../pretrained/SiamFC_new/'+name_head+'_'+str(squeeze_iter)+'_'+str(step)+'_'+str(turn)+'_'+str(i)+'.pth'
            tracker_name = name_head + '_'+str(squeeze_iter)+'_'+str(step)+'_'+str(turn)+'_' + str(i)
            for experiment in experiments:
                tracker_infos.append([model_path, tracker_name, experiment, squeeze_rate, step])
    p = Pool(processes=4)
    for tracker_info in tracker_infos:
        p.apply_async(evaluate, args=tuple(tracker_info))
    p.close()
    p.join()


def best_result_otb2013(result_dir, name_key_word):
    tracker_names = os.listdir(os.path.join(result_dir, 'OTB2013'))  # 获取OTB2015路径下的所有结果

    tracker_names = [x for x in tracker_names if name_key_word in x]  # 对特定的结果进行筛选
    # tracker_names = [x for x in tracker_names if '.' in x]

    tracker_results = []
    for tracker_name in tracker_names:
        tracker_dir = os.path.join(result_dir, 'OTB2013', tracker_name, 'performance.json')
        with open(tracker_dir) as fp:
            result = json.load(fp)
            tracker_results.append(result)

    # 从tracker_result中解析所有跟踪器的结果，并绘制相应的曲线，或者图表。
    # otb2015的结果中，主要包含三个重要的信息，分别是success_score, precision_score, success_rate.将这三个信息读取，并存储为list
    successScore = []
    precisionScore = []
    successRate = []
    for result in tracker_results:
        successScore.append(result[list(result.keys())[0]]['overall']['success_score'])
        precisionScore.append(result[list(result.keys())[0]]['overall']['precision_score'])
        successRate.append(result[list(result.keys())[0]]['overall']['success_rate'])
    # 根据结果绘制条形图

    # 根据得分对所有模型下的跟踪结果进行排序。
    scores = [0.4*a+0.4*b+0.2*c for a, b, c in zip(successScore, precisionScore, successRate)]
    scores_decay_index = np.fromiter(scores, dtype=np.float).argsort()[::-1]
    successScore = [successScore[x] for x in scores_decay_index]
    precisionScore = [precisionScore[x] for x in scores_decay_index]
    successRate = [successRate[x] for x in scores_decay_index]
    tracker_names = [tracker_names[x] for x in scores_decay_index]

    model_name = os.path.join(r'../pretrained/SiamFC_new/', tracker_names[0])+'.pth'
    return model_name, successScore[0], precisionScore[0], successRate[0]