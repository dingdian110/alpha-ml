import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='gc')
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--dataset', type=str, default='pc4')
parser.add_argument('--start_runid', type=int, default=0)
args = parser.parse_args()

if args.mode == 'master':
    project_folder = '/home/thomas/PycharmProjects/alpha-ml'
elif args.mode == 'daim213':
    project_folder = '/home/liyang/codes/alpha-ml'
elif args.mode == 'gc':
    project_folder = '/home/contact_ds3lab/testinstall/alpha-ml'
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def plot(dataset, rep_num, start_id):
    task_id = 'exp_1_evaluation_500'
    mth_list = ['avg_smac', 'cmab_ts_smac', 'smac',
                'softmax_mab_1_1.0000_smac', 'ucb_mab_1_0.0000_smac', 'mm_bandit_3_smac']
    optimizer_algos = ['baseline_2', 'cmab_ts', 'smbo', 'rl_2_1', 'rl_3_0', 'mono_smbo_3_0']
    assert len(mth_list) == len(optimizer_algos)
    exp_result = dict()
    for idx, mth in enumerate(mth_list):
        exp_result[mth] = list()
        perfs = list()
        tmp_d = dataset.split('_')[0]
        # Extract validation performance.
        for id in range(rep_num):
            file_id = project_folder + '/data/%s/%s_%s_%d_%s.data' % (tmp_d, dataset, task_id, id, mth)
            print(file_id)
            with open(file_id, 'rb') as f:
                data = pickle.load(f)
            perf_t = np.max(data['perfs'])
            print(mth, data['time_cost'][-1], perf_t)
            perfs.append(perf_t)
        assert len(perfs) == rep_num
        mean_perf = np.mean(perfs)
        print('='*10, mth, mean_perf)
        exp_result[mth].append(mean_perf)

        # Extract test performance.
        with open(project_folder + '/data/%s/%s_test_result_%s_%s_%d_%d.pkl' %
                  (tmp_d, dataset, optimizer_algos[idx], task_id, rep_num, start_id), 'rb') as f:
            test_data = pickle.load(f)
        assert len(test_data.values()) == rep_num
        exp_result[mth].append(np.mean(test_data.values()))
    print('='*50)
    print(exp_result)
    print('='*50)


if __name__ == "__main__":
    plot(args.dataset, args.rep, args.start_runid)
