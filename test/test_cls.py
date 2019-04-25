import sys
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
sns.set_style(style='whitegrid')

plt.rc('text', usetex=True)
plt.rc('font', size=12.0, family='Times New Roman')
plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'black'
plt.rc('legend', **{'fontsize': 12})

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213'], default='master')
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--run_count', type=int, default=100)
parser.add_argument('--datasets', type=str, default='iris')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def test_configspace():
    from alphaml.engine.components.components_manager import ComponentsManager
    from alphaml.engine.components.models.classification import _classifiers

    # print(_classifiers)
    # for item in _classifiers:
    #     name, cls = item, _classifiers[item]
    #     print(cls.get_hyperparameter_search_space())
    cs = ComponentsManager().get_hyperparameter_search_space(3)
    print(cs.sample_configuration(5))
    # print(cs.get_default_configuration())


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    rep_num = args.rep
    run_count = args.run_count
    datasets = args.datasets.split(',')
    print(rep_num, run_count, datasets)

    for dataset in datasets:
        for run_id in range(rep_num):
            for optimizer in ['smbo']:
                task_format = dataset + '_alpha_3_%d'
                X, y, _ = load_data(dataset)
                cls = Classifier(
                    include_models=['gaussian_nb', 'adaboost', 'random_forest', 'k_nearest_neighbors', 'gradient_boosting'],
                    optimizer=optimizer).fit(DataManager(X, y), metric='accuracy', runcount=run_count, task_name=task_format % run_id)
                print(cls.predict(X))


def plot(dataset, rep_num):
    color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    mth_list = ['smac', 'ts_smac']
    lw = 2
    ms = 4
    me = 10

    color_dict, marker_dict = dict(), dict()
    for index, mth in enumerate(mth_list):
        color_dict[mth] = color_list[index]
        marker_dict[mth] = markers[index]

    fig, ax = plt.subplots(1)
    handles = list()
    x_num = 100

    for mth in mth_list:
        perfs = list()
        for id in range(rep_num):
            file_id = 'data/%s_%d_%s.data' % (dataset, id, mth)
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            perfs.append(data['perfs'])
        perfs = np.mean(perfs, axis=0)
        print(max(perfs), max(perfs[:27]))
        x_num = len(perfs)
        ax.plot(list(range(x_num)), perfs, label=mth, lw=lw, color=color_dict[mth],
                marker=marker_dict[mth], markersize=ms, markevery=me)
        line = mlines.Line2D([], [], color=color_dict[mth], marker=marker_dict[mth],
                             markersize=ms, label=r'\textbf{%s}' % mth.replace("_", "\\_"))
        handles.append(line)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_num // 10))
    legend = ax.legend(handles=handles, loc='best')
    ax.set_xlabel('\\textbf{Iteration}', fontsize=15)
    ax.set_ylabel('\\textbf{Validation accuracy}', fontsize=15)
    plt.show()


def debug(dataset, id):
    for mth in ['smac', 'ts_smac']:
        file_id = 'data/%s_%d_%s.data' % (dataset, id, mth)
        with open(file_id, 'rb') as f:
            data = pickle.load(f)

        count_dict = dict()
        perf_dict = dict()
        for config, perf in zip(data['configs'], data['perfs']):
            est = config['estimator']
            if est not in count_dict:
                count_dict[est] = 0
                perf_dict[est] = list()
            count_dict[est] += 1
            perf_dict[est].append(perf)
        print('='*30, mth, '='*30)
        print(count_dict)

        max_id = np.argmax(data['perfs'])
        print(data['configs'][max_id])

        for key in sorted(perf_dict.keys()):
            print(key, np.mean(perf_dict[key]), np.std(perf_dict[key]))
        if mth == 'ts_smac':
            print(data['ts_params'])
            print(data['ts_cnts'])
        print(perf_dict)


if __name__ == "__main__":
    test_cash_module()
    # plot('svmguide4', 1)
