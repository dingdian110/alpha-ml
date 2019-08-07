import sys
import time
import pickle
import argparse
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
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--run_count', type=int, default=200)
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--datasets', type=str, default='glass')
parser.add_argument('--task_id', type=str, default='test')
parser.add_argument('--opt_algo', type=str, default='baseline_avg')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)

from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier
from alphaml.datasets.cls_dataset.dataset_loader import load_data

from sklearn.model_selection import train_test_split


def get_seeds(dataset, rep_num):
    # Map the dataset to a fixed integer.
    dataset_id = int(''.join([str(ord(c)) for c in dataset[:6] if c.isalpha()])) % 100000
    np.random.seed(dataset_id)
    return np.random.random_integers(10000, size=rep_num)


def test_cash_module():
    rep_num = args.rep
    run_count = args.run_count
    start_id = args.start_runid
    datasets = args.datasets.split(',')
    optimizer_algos = args.opt_algo.split(',')
    task_id = args.task_id
    print(rep_num, run_count, datasets, optimizer_algos, task_id)

    result = dict()
    for dataset in datasets:
        seeds = get_seeds(dataset, rep_num)
        for run_id in range(start_id, rep_num):
            task_name = dataset + '_%s_%d_%d' % (task_id, run_count, run_id)
            seed = seeds[run_id]

            # Dataset partition.
            X, y, _ = load_data(dataset)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            dm = DataManager(X_train, y_train)

            # Test each optimizer algorithm:
            for optimizer in optimizer_algos:
                # Parse the parameters for each optimizer.
                mode = 2
                eta, r = 2, 2
                if optimizer.startswith('baseline'):
                    optimizer, mode = optimizer.split('_')
                    mode = 1 if mode == 'rand' else 2
                if optimizer.startswith('sh'):
                    if len(optimizer.split('_')) == 2:
                        optimizer, eta = optimizer.split('_')
                        eta = float(eta)
                    else:
                        raise ValueError('Wrong SH params!')
                if optimizer.startswith('rl'):
                    if len(optimizer.split('_')) == 3:
                        _, mode, eta = optimizer.split('_')
                        mode = int(mode)
                        optimizer = 'rl_smbo'
                    else:
                        raise ValueError('Wrong SH params!')
                if optimizer.startswith('ts_smbo'):
                    mode = 1
                    if len(optimizer.split('_')) == 3:
                        _, _, mode = optimizer.split('_')
                        mode = int(mode)
                        optimizer = 'ts_smbo'
                if optimizer.startswith('mcmc_ts'):
                    _, _, mode, eta, r = optimizer.split('_')
                    mode = int(mode)
                    eta = int(eta)
                    r = int(r)
                    optimizer = 'mcmc_ts_smbo'

                if optimizer.startswith('ucb_smbo'):
                    mode = 1
                    if len(optimizer.split('_')) == 3:
                        _, _, mode = optimizer.split('_')
                        mode = int(mode)
                        optimizer = 'ucb_smbo'

                if optimizer.startswith('mono_smbo'):
                    mode = 2
                    if len(optimizer.split('_')) == 4:
                        _, _, mode, r = optimizer.split('_')
                        mode, r = int(mode), int(r)
                        eta = 10
                        optimizer = 'mono_smbo'

                print('Test %s optimizer => %s' % (optimizer, task_name))

                # Construct the AutoML classifier.
                cls = Classifier(optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=run_count,
                    task_name=task_name, update_mode=mode, eta=eta, r=r, param=eta)
                acc = cls.score(X_test, y_test)
                key_id = '%s_%d_%d_%s' % (dataset, run_count, run_id, optimizer)
                result[key_id] = acc

            # Display and save the test result.
            print(result)
            dataset_id = dataset.split('_')[0]
            with open('data/%s/%s_test_result_%s_%d_%d_%d.pkl' %
                              (dataset_id, dataset_id, task_id, run_count, rep_num, start_id), 'wb') as f:
                pickle.dump(result, f)


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
