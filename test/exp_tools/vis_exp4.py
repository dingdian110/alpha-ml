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
    task_id = "exp4_runtime_0"
    optimizer_algos = ['smbo', 'mono_smbo_4']
    exp_result = dict()
    tmp_d = dataset.split('_')[0]

    for algo in optimizer_algos:
        # Extract test performance.
        file_id = '/data/%s/%s_test_result_%s_%s_%d_%d.pkl' % (tmp_d, dataset, algo, task_id, rep_num, start_id)
        print(file_id)
        try:
            with open(project_folder + file_id, 'rb') as f:
                test_data = pickle.load(f)
                values = np.array(list(test_data.values()))
            assert len(values) == rep_num
            exp_result[algo].append(np.mean(values, axis=0))
        except EOFError:
            pass

    print('='*50)
    for item, values in exp_result.items():
        print(item.ljust(15, ' '), ['%.2f' % (100*val) for val in values])
    print('='*50)


if __name__ == "__main__":
    plot(args.dataset, args.rep, args.start_runid)
