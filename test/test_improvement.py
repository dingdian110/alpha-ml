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
parser.add_argument('--rep', type=int, default=50)
parser.add_argument('--run_count', type=int, default=200)
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--datasets', type=str, default='glass')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)


rep_num = args.rep
run_count = args.run_count
start_run = args.start_runid
datasets = args.datasets.split(',')

print(rep_num, run_count, datasets)


def test_hyperspace():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from alphaml.utils.constants import MAX_INT

    try:
        for dataset in datasets:
            for run_id in range(start_run, rep_num):
                X, y, _ = load_data(dataset)
                dm = DataManager(X, y)
                seed = np.random.random_integers(MAX_INT)

                for update_mode in [2, 3]:
                    task_format = dataset + '_mode_%d_%d' % (update_mode, run_id)
                    cls = Classifier(optimizer='ts_smbo', seed=seed).fit(
                        dm, metric='accuracy', runcount=run_count,
                        task_name=task_format, update_mode=update_mode)
                    print(cls.predict(X))
    except Exception as e:
        print(e)
        print('Exit!')


def plot():
    dataset = datasets[0]
    color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    mth_list = [1, 2, 3]
    lw = 2
    ms = 4
    me = 10

    color_dict, marker_dict = dict(), dict()
    for index, mth in enumerate(mth_list):
        color_dict[mth] = color_list[index]
        marker_dict[mth] = markers[index]

    fig, ax = plt.subplots(1)
    handles = list()
    x_num = 20

    for mth in mth_list:
        perfs = list()
        for id in range(rep_num):
            file_id = 'data/%s/%s_mode_%d_%d_%s.data' % (dataset, dataset, mth, id, 'smac')
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            perfs.append(data['perfs'])
        perfs = np.mean(perfs, axis=0)
        print(mth, max(perfs))
        x_num = len(perfs)
        ax.plot(list(range(x_num)), perfs, label=mth, lw=lw, color=color_dict[mth],
                marker=marker_dict[mth], markersize=ms, markevery=me)
        line = mlines.Line2D([], [], color=color_dict[mth], marker=marker_dict[mth],
                             markersize=ms, label=r'\textbf{mode-%d}' % mth)
        handles.append(line)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_num // 10))
    legend = ax.legend(handles=handles, loc='best')
    ax.set_xlabel('\\textbf{Iteration}', fontsize=15)
    ax.set_ylabel('\\textbf{Validation accuracy}', fontsize=15)
    plt.show()


if __name__ == "__main__":
    test_hyperspace()
    # plot()
