import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='gc')
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--dataset', type=str, default='pc4')
args = parser.parse_args()

if args.mode == 'master':
    project_folder = '/home/thomas/PycharmProjects/alpha-ml'
elif args.mode == 'daim213':
    project_folder = '/home/liyang/codes/alpha-ml'
elif args.mode == 'gc':
    project_folder = '/home/contact_ds3lab/testinstall/alpha-ml'
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def plot(dataset, rep_num):
    task_id = 'exp_1_evaluation_500'
    mth_list = ['avg_smac', 'cmab_ts_smac', 'smac',
                'mm_bandit_3_smac', 'softmax_mab_1_1.0000_smac', 'ucb_mab_1_0.0000_smac']
    for mth in mth_list:
        perfs = list()
        tmp_d = dataset.split('_')[0]
        for id in range(rep_num):
            file_id = project_folder + '/data/%s/%s_%s_%d_%s.data' % (tmp_d, dataset, task_id, id, mth)
            print(file_id)
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            perf_t = np.max(data['perfs'])
            print(mth, data['time_cost'][-1], perf_t)
            perfs.append(perf_t)
        print('='*10, mth, np.mean(perfs))


if __name__ == "__main__":
    plot(args.dataset, args.rep)
