import sys
import argparse
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
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
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--run_count', type=int, default=200)
parser.add_argument('--datasets', type=str, default='iris')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)


rep_num = args.rep
run_count = args.run_count
datasets = args.datasets.split(',')
algo_list = ['adaboost', 'random_forest', 'k_nearest_neighbors', 'gradient_boosting',
             'decision_tree', 'extra_trees', 'lda', 'liblinear_svc',
             'libsvm_svc', 'logistic_regression', 'sgd', 'xgboost']
print(rep_num, run_count, datasets)


def get_seeds(dataset, rep_num):
    # Map the dataset to a fixed integer.
    dataset_id = int(''.join([str(ord(c)) for c in dataset[:6] if c.isalpha()])) % 100000
    np.random.seed(dataset_id)
    return np.random.random_integers(10000, size=rep_num)


def test_no_free_lunch():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    for dataset in datasets:
        seeds = get_seeds(dataset, rep_num)
        for run_id in range(rep_num):
            seed = seeds[run_id]

            # Dataset partition.
            X, y, _ = load_data(dataset)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            dm = DataManager(X_train, y_train)
            for algo in algo_list:
                for optimizer in ['smbo']:
                    task_format = dataset + '_' + algo + '_%d_%d'
                    cls = Classifier(
                        include_models=[algo], optimizer=optimizer, seed=seed).fit(
                        dm, metric='accuracy', runcount=run_count, task_name=task_format % (run_count, run_id))
                    print(cls.predict(X))


def plot():
    dataset = datasets[0]
    color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    mth_list = algo_list
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
            file_id = 'data/%s_%d_%s.data' % (dataset+mth, id, 'smac')
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            perfs.append(data['perfs'])
        perfs = np.mean(perfs, axis=0)
        print(mth, max(perfs))
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


if __name__ == "__main__":
    test_no_free_lunch()
    # plot()
