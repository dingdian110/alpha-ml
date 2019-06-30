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
args = parser.parse_args()

sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
from alphaml.engine.optimizer.reward_models.mcmc_model import MCMCModel
# sys.path.append('/home/liyang/codes/alpha-ml')
data_folder = '/home/thomas/PycharmProjects/alpha-ml/data/'


def sample_curve(model, xlim):
    preds = list()
    model_samples = model.get_burned_in_samples()
    theta_idx = np.random.randint(0, model_samples.shape[0], 1)[0]
    theta = model_samples[theta_idx, :]

    params, sigma = model.curve_model.split_theta(theta)
    for i in range(1, xlim + 1):
        predictive_mu = model.curve_model.function(i, *params)
        preds.append(predictive_mu)
    return np.asarray(preds)


def weibull_examples():
    y = np.array([0.5, 0.5])
    x = np.array(list(range(1, len(y) + 1)))
    model = MCMCModel()
    model.fit_mcmc(x, y)
    curve_num = 50
    x_lim = 100
    x_val = np.array(list(range(1, x_lim + 1)))
    for _ in range(curve_num):
        y_val = sample_curve(model, x_lim)
        plt.plot(x_val, y_val)
    plt.ylim(0.5, 1)
    plt.xlabel('\\textbf{Steps}', fontsize=15)
    plt.ylabel('\\textbf{Function values}', fontsize=15)
    plt.savefig(data_folder + "weibull_examples.pdf")
    plt.show()


def fit_example():
    # pc4, random_forest.
    y = [0.8846153846153846, 0.8846153846153846, 0.8888888888888888, 0.8931623931623932, 0.8931623931623932,
         0.8931623931623932, 0.9145299145299145, 0.9145299145299145, 0.9145299145299145]
    y.extend([0.9145299145299145] * 10)

    y = np.asarray(y)
    x = np.array(list(range(1, len(y) + 1)))
    model = MCMCModel()
    model.fit_mcmc(x, y)
    x_pred = list(range(1, 50))
    y_pred = list()
    y_sigma = list()
    for t in x_pred:
        mu, sigma = model.predict(t)
        y_pred.append(mu)
        y_sigma.append(sigma)

    from matplotlib import pyplot as plt
    plt.plot(x, y, color='blue')
    y_pred = np.array(y_pred)
    y_sigma = np.array(y_sigma)
    plt.plot(x_pred, y_pred, color='red')
    plt.fill_between(x_pred, y_pred - y_sigma, y_pred + y_sigma, facecolor='green', alpha=0.2)
    print(y_sigma[len(x)+1:])
    plt.ylim(0.8, 1)
    plt.xlabel('\\textbf{Steps}', fontsize=15)
    plt.ylabel('\\textbf{Rewards}', fontsize=15)
    plt.show()


def transform(data, run_count):
    transformed_data = list()
    flag = 0.
    for item in data:
        flag = item if item > flag else flag
        transformed_data.append(flag)
    left_num = run_count + 1 - len(transformed_data)
    if len(transformed_data) < run_count:
        transformed_data.extend([flag] * left_num)
    else:
        transformed_data = transformed_data[:run_count]
    return transformed_data


def posterior_example(dataset='glass'):
    color_list = ['purple', 'royalblue', 'green', 'red', 'brown', 'orange', 'yellowgreen']
    markers = ['s', '^', '2', 'o', 'v', 'p', '*']
    mth_list = ['random_forest:smac']
    lw = 2
    ms = 4
    me = 10
    algo = 'random_forest'
    run_count = 200
    rep = 5
    train_num = 10
    color_dict, marker_dict = dict(), dict()
    for index, mth in enumerate(mth_list):
        color_dict[mth] = color_list[index]
        marker_dict[mth] = markers[index]

    fig, ax = plt.subplots(1)
    handles = list()
    perfs = list()

    for id in range(rep):
        file_id = data_folder + '%s/%s_%s_200_%d_smac.data' % (dataset, dataset, algo, id)
        with open(file_id, 'rb') as f:
            data = pickle.load(f)
        perfs.append(transform(data['perfs'], run_count))
    raw_perfs = perfs
    perfs = np.mean(perfs, axis=0)
    x_num = len(perfs)
    ax.plot(list(range(1, train_num+1)), perfs[:train_num], label='random_forest', lw=lw, color=color_list[0],
            marker=markers[0], markersize=ms, markevery=me)
    ax.plot(list(range(train_num+1, x_num+1)), perfs[train_num:],
            label='random_forest', color='gray', linestyle="--")
    line = mlines.Line2D([], [], color=color_list[0], marker=markers[0],
                         markersize=ms, label=r'\textbf{smac}')
    handles.append(line)
    print('finish smac!')

    # MCMC model evaluation.
    mcmc_res = list()
    for iter, perf in enumerate(raw_perfs):
        print('finish iteration: %d' % iter)
        y = np.asarray(perf[:train_num])
        x = np.array(list(range(1, len(y) + 1)))
        model = MCMCModel()
        model.fit_mcmc(x, y)
        tmp_res = list()
        for t in range(1, run_count+1):
            mu, _ = model.predict(t)
            tmp_res.append(mu)
        mcmc_res.append(tmp_res)
    mcmc_res = np.asarray(mcmc_res)
    mcmc_y = np.mean(mcmc_res, axis=0)
    ax.plot(list(range(1, 1+train_num)), mcmc_y[:train_num], lw=1, color=color_list[1], linestyle="--")
    ax.plot(list(range(train_num+1, train_num+11)), mcmc_y[train_num:train_num+10], label='mcmc', lw=2, color=color_list[1],
            marker=markers[1], markersize=ms, markevery=me)
    line = mlines.Line2D([], [], color=color_list[1], marker=markers[1],
                         markersize=ms, label=r'\textbf{MCMC Inference}')
    handles.append(line)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_num // 10))
    legend = ax.legend(handles=handles, loc='best')
    plt.axvline(train_num, linestyle="--", color="brown", lw=1)
    ax.set_xlabel('\\textbf{Steps}', fontsize=15)
    ax.set_ylabel('\\textbf{Rewards}', fontsize=15)
    plt.savefig(data_folder + "posterior_examples_%d.pdf" % train_num)
    plt.show()


if __name__ == "__main__":
    # weibull_examples()
    # fit_example()
    posterior_example()
