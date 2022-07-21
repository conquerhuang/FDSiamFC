from matplotlib import pyplot as plt
import os
from glob import glob
import json
import numpy as np


def result_analysis_otb2015(result_dir):
    tracker_names = os.listdir(os.path.join(result_dir, 'OTB2015'))  # 获取OTB2015路径下的所有结果

    tracker_names = [x for x in tracker_names if 'FDSiam_8l_' in x]  # 对特定的结果进行筛选
    # tracker_names = [x for x in tracker_names if '.' in x]

    tracker_results = []
    for tracker_name in tracker_names:
        tracker_dir = os.path.join(result_dir, 'OTB2015', tracker_name, 'performance.json')
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

    # 根据得分对所有模型下的跟踪结果进行排序。
    scores = [0.4*a+0.4*b+0.2*c for a, b, c in zip(successScore, precisionScore, successRate)]
    scores_decay_index = np.fromiter(scores, dtype=np.float).argsort()[::-1]
    successScore = [successScore[x] for x in scores_decay_index]
    precisionScore = [precisionScore[x] for x in scores_decay_index]
    successRate = [successRate[x] for x in scores_decay_index]
    tracker_names = [tracker_names[x] for x in scores_decay_index]

    # 根据结果绘制条形图
    x = np.arange(1, len(successScore)+1)
    bar_wideth = 0.2
    p1 = plt.bar(x, successScore, bar_wideth, label='success score', tick_label=tracker_names)
    p2 = plt.bar(x+bar_wideth, precisionScore, bar_wideth, label='precision score')
    p3 = plt.bar(x+bar_wideth+bar_wideth, successRate, bar_wideth, label='success rate')
    plt.xticks(rotation=90)
    plt.legend([p1, p2, p3], ['success score', 'precision score', 'success rate'])
    plt.show()
    pass
    pass

def result_analysis_otb2013(result_dir):
    tracker_names = os.listdir(os.path.join(result_dir, 'OTB2013'))  # 获取OTB2015路径下的所有结果

    tracker_names = [x for x in tracker_names if 'SiamFC_' in x]  # 对特定的结果进行筛选
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

    x = np.arange(1, len(successScore)+1)
    bar_wideth = 0.2
    p1 = plt.bar(x, successScore, bar_wideth, label='success score', tick_label=tracker_names)
    p2 = plt.bar(x+bar_wideth, precisionScore, bar_wideth, label='precision score')
    p3 = plt.bar(x+bar_wideth+bar_wideth, successRate, bar_wideth, label='success rate')
    plt.xticks(rotation=90)
    plt.legend([p1, p2, p3], ['success score', 'precision score', 'success rate'])
    plt.show()
    pass
    pass


def result_analysis_tb50(result_dir):
    tracker_names = os.listdir(os.path.join(result_dir, 'OTBtb50'))  # 获取OTB2015路径下的所有结果

    tracker_results = []
    for tracker_name in tracker_names:
        tracker_dir = os.path.join(result_dir, 'OTBtb50', tracker_name, 'performance.json')
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

    # 根据得分对所有模型下的跟踪结果进行排序。
    scores = [0.4*a+0.4*b+0.2*c for a, b, c in zip(successScore, precisionScore, successRate)]
    scores_decay_index = np.fromiter(scores, dtype=np.float).argsort()[::-1]
    successScore = [successScore[x] for x in scores_decay_index]
    precisionScore = [precisionScore[x] for x in scores_decay_index]
    successRate = [successRate[x] for x in scores_decay_index]
    tracker_names = [tracker_names[x] for x in scores_decay_index]

    # 根据结果绘制条形图
    x = np.arange(1, len(successScore)+1)
    bar_wideth = 0.2
    p1 = plt.bar(x, successScore, bar_wideth, label='success score', tick_label=tracker_names)
    p2 = plt.bar(x+bar_wideth, precisionScore, bar_wideth, label='precision score')
    p3 = plt.bar(x+bar_wideth+bar_wideth, successRate, bar_wideth, label='success rate')
    plt.xticks(rotation=90)
    plt.legend([p1, p2, p3], ['success score', 'precision score', 'success rate'])
    plt.show()


def result_analysis_vot2016(result_dir):
    tracker_names = os.listdir(os.path.join(result_dir, 'VOT2016'))  # 获取OTB2015路径下的所有结果

    tracker_results = []
    for tracker_name in tracker_names:
        tracker_dir = os.path.join(result_dir, 'VOT2016', tracker_name, 'performance.json')
        with open(tracker_dir) as fp:
            result = json.load(fp)
            tracker_results.append(result)

    # 从tracker_result中解析所有跟踪器的结果，并绘制相应的曲线，或者图表。
    # otb2015的结果中，主要包含三个重要的信息，分别是success_score, precision_score, success_rate.将这三个信息读取，并存储为list
    accuracy = []
    robustness = []
    for result in tracker_results:
        accuracy.append(result[list(result.keys())[0]]['accuracy'])
        robustness.append(result[list(result.keys())[0]]['robustness']/50.)

    # 根据得分对所有模型下的跟踪结果进行排序。
    scores = [0.5*a+0.5*b for a, b in zip(accuracy, robustness)]
    scores_decay_index = np.fromiter(scores, dtype=np.float).argsort()[::-1]
    accuracy = [accuracy[x] for x in scores_decay_index]
    robustness = [robustness[x] for x in scores_decay_index]
    tracker_names = [tracker_names[x] for x in scores_decay_index]

    # 根据结果绘制条形图
    x = np.arange(1, len(accuracy)+1)
    bar_wideth = 0.2
    p1 = ax1 = plt.bar(x, accuracy, bar_wideth, label='success score', tick_label=tracker_names)
    p2 = plt.bar(x+bar_wideth, robustness, bar_wideth, label='precision score(1/50)')
    plt.xticks(rotation=90)
    plt.legend([p1, p2], ['success score', 'precision score(1/50)'])
    plt.show()


def main():
    # got10k结果路径
    result_dir = r'./reports'

    result_analysis_otb2015(result_dir)
    # result_analysis_otb2013(result_dir)
    # result_analysis_tb50(result_dir)
    # result_analysis_vot2016(result_dir)


if __name__ == '__main__':
    main()


