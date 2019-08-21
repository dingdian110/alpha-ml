import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.mlab as mlab
import seaborn as sns
import argparse

plt.switch_backend('agg')
sns.set_style(style='whitegrid')

plt.rc('font', size=12.0)
plt.rcParams['figure.figsize'] = (8.0, 4.0)
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'black'
plt.rc('legend', **{'fontsize': 12})

data_folder = '/home/contact_ds3lab/testinstall/alpha-ml/data/'


parser = argparse.ArgumentParser()
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--dataset', type=str, default='elevators')
parser.add_argument('--mth', type=str, default='0,1')
args = parser.parse_args()


def transform(data):
    transformed_data = list()
    for item in data:
        if len(transformed_data) == 0:
            val = item
        else:
            val = transformed_data[-1]
        if item > val:
            val = item
        transformed_data.append(val)
    return np.asarray(transformed_data)


def fx(x, func):
    perf_list = []
    for xt in x:
        last_p = 0.
        for t, p in func:
            if t > xt:
                break
            last_p = p
        perf_list.append(last_p)
    return np.asarray(perf_list)


def plot(dataset, rep_num, mths):
    b = 3600
    task_id = 'exp3_cost_aware'
    task_id = '%s_%d_0' % (task_id, b)
    color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    mth_sets= ['mm_bandit_3_smac', 'mm_bandit_4_smac', 'smac']
    dis_names = ['RB', 'RB*', 'SMAC']
    mth_list = [mth_sets[item] for item in mths]
    print(mth_list)

    lw = 2
    ms = 4
    me = 100

    color_dict, marker_dict = dict(), dict()
    for index, mth in enumerate(mth_list):
        color_dict[mth] = color_list[index]
        marker_dict[mth] = markers[index]

    fig, ax = plt.subplots(1)
    handles = list()

    for idx, mth in enumerate(mth_list):
        perfs = list()
        x = np.linspace(0, b, num=1000)
        tmp_d = dataset.split('_')[0]
        for id in range(rep_num):
            file_id = data_folder + '%s/%s_%s_%d_%s.data' % (tmp_d, dataset, task_id, id, mth)
            print(file_id)
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            stats = np.array([data['time_cost'], transform(data['perfs'])]).T
            fs = fx(x, stats)
            perfs.append(fs)

        print(np.array(perfs).shape)
        perfs = np.mean(perfs, axis=0)
        print(mth, max(perfs), np.argmax(perfs))
        x_num = len(perfs)
        method_name = dis_names[idx]
        # method_name = 'Test' if mth == 'ts_smac' else 'Auto-Sklearn'
        ax.plot(x, perfs, label=mth, lw=lw, color=color_dict[mth],
                marker=marker_dict[mth], markersize=ms, markevery=me)
        line = mlines.Line2D([], [], color=color_dict[mth], marker=marker_dict[mth],
                             markersize=ms, label=method_name)
        handles.append(line)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    legend = ax.legend(handles=handles)
    ax.set_xlabel('Time (Seconds)', fontsize=15)
    ax.set_ylabel('Validation accuracy', fontsize=15)
    ax.set_ylim(.8, .93)

    plt.savefig("./exp3_%s.pdf" % dataset)
    # plt.show()


if __name__ == "__main__":
    mths = [int(item) for item in args.mth.split(',')]
    plot(args.dataset, args.rep, mths=mths)
