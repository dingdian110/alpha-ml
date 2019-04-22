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


def test_load_data():
    with open('data/random_forest.data', 'rb') as f:
        data = pickle.load(f)
        print(data['perfs'])
        print(data['time_cost'])


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
    plot('svmguide4', 1)
