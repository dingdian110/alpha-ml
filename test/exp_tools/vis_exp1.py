import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='gc')
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--dataset', type=str, default='pc4')
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--mth', type=str, default='0')
args = parser.parse_args()

if args.mode == 'master':
    project_folder = '/home/thomas/PycharmProjects/alpha-ml'
elif args.mode == 'daim213':
    project_folder = '/home/liyang/codes/alpha-ml'
elif args.mode == 'gc':
    project_folder = '/home/contact_ds3lab/testinstall/alpha-ml'
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def generate_str(missing_flag, data_res):
    max_val = np.max(data_res)
    str_list = list()
    for idx, val in enumerate(data_res):
        if val == max_val:
            str_list.append('$\\bm{%.2f}$' % (100 * val))
        else:
            str_list.append('%.2f' % (100 * val))
    if len(missing_flag) > len(data_res):
        for idx, flag in enumerate(missing_flag):
            if flag:
                str_list.insert(idx, 'x')
    string = ' & '.join(str_list)
    return string


def plot(dataset, rep_num, start_id):
    algo_ids = [int(id) for id in args.mth.split(',')]
    task_id = 'exp_1_evaluation_500'
    mth_list_set = ['avg_smac', 'smac', 'ucb_mab_1_0.0000_smac', 'cmab_ts_smac',
                'softmax_mab_1_1.0000_smac', 'mm_bandit_3_smac', 'mm_bandit_3_tpe']
    optimizer_algos = ['baseline_2', 'cmab_ts', 'smbo', 'rl_2_1', 'rl_3_0', 'mono_smbo_3_0', 'mono_tpe_smbo']
    assert len(mth_list_set) == len(optimizer_algos)
    mth_list = [mth_list_set[id] for id in algo_ids]
    exp_result = dict()
    for idx, mth in enumerate(mth_list_set):
        if not mth in mth_list:
            continue
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
        # assert len(perfs) == rep_num
        mean_perf = np.mean(perfs)
        print('='*10, mth, mean_perf)
        exp_result[mth].append(mean_perf)

        # Extract test performance.
        file_id = '/data/%s/%s_test_result_%s_%s_%d_%d.pkl' % (tmp_d, dataset,
            optimizer_algos[idx], task_id, rep_num, start_id)
        print(file_id)
        try:
            with open(project_folder + file_id, 'rb') as f:
                test_data = pickle.load(f)
                values = list(test_data.values())
            # assert len(values) == rep_num
            exp_result[mth].append(np.mean(values))
        except EOFError:
            pass

    print('='*50)
    for mth in mth_list:
        item = mth
        values = exp_result[mth]
        print(item.ljust(30, ' '), ['%.2f' % (100*val) for val in values])
    print('='*50)

    val_res, test_res = list(), list()
    missing_flag = [False] * len(mth_list)

    for mth in mth_list:
        if len(exp_result[mth]) < 2:
            missing_flag[mth] = True
        else:
            val1, val2 = exp_result[mth]
            val_res.append(val1)
            test_res.append(val2)

    val_strings = generate_str(missing_flag, val_res)
    test_strings = generate_str(missing_flag, test_res)
    print(val_strings)
    print(test_strings)
    print('Latex code: ', '%s & %s & \n & %s \\\\' % (dataset.upper(), val_strings, test_strings))


if __name__ == "__main__":
    plot(args.dataset, args.rep, args.start_runid)
