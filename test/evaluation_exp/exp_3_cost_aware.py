import os
import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='master')
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--run_count', type=int, default=500)
parser.add_argument('--B', type=int, default=3600)
parser.add_argument('--datasets', type=str, default='pc4')
args = parser.parse_args()

if args.mode == 'master':
    project_folder = '/home/thomas/PycharmProjects/alpha-ml'
elif args.mode == 'daim213':
    project_folder = '/home/liyang/codes/alpha-ml'
elif args.mode == 'gc':
    project_folder = '/home/contact_ds3lab/testinstall/alpha-ml'
else:
    raise ValueError('Invalid mode: %s' % args.mode)
sys.path.append(project_folder)

from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier
from alphaml.datasets.cls_dataset.dataset_loader import load_data
from sklearn.model_selection import train_test_split


def get_seeds(dataset, rep_num):
    # Map the dataset to a fixed integer.
    dataset_id = int(''.join([str(ord(c)) for c in dataset[:6] if c.isalpha()])) % 100000
    np.random.seed(dataset_id)
    return np.random.random_integers(10000, size=rep_num)


def load_infos(dataset, task_id, run_count, id, mth):
    data_folder = project_folder + '/data/'
    tmp_d = dataset.split('_')[0]
    file_id = data_folder + '%s/%s_%s_%d_%d_%s.data' % (tmp_d, dataset, task_id, run_count, id, mth)
    with open(file_id, 'rb') as f:
        data = pickle.load(f)
    configs, perfs = data['configs'], data['perfs']
    assert len(configs) == len(perfs)
    return configs, perfs


def test_exp3_cost_aware():
    rep_num = args.rep
    run_count = args.run_count
    B = args.B
    if B > 0:
        run_count = 0

    start_id = args.start_runid
    datasets = args.datasets.split(',')
    print(rep_num, run_count, datasets)
    task_id = "exp3_cost_aware"

    for dataset in datasets:
        dataset_id = dataset.split('_')[0]
        result_dir = 'data/'+dataset_id
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        # Dataset partition.
        X, y, _ = load_data(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                            stratify=y)
        dm = DataManager(X_train, y_train)

        # optimizer_algos = ['cmab_ts', 'mono_smbo_3', 'mono_smbo_4', 'smbo', 'tpe']
        # optimizer_algos = ['mono_smbo_3', 'cmab_ts']
        optimizer_algos = ['mono_smbo_4', 'smbo', 'cmab_ts', 'tpe']
        # Test each optimizer algorithm:
        runcount_dict = dict()
        tpe_runcount = 0.

        for opt_algo in optimizer_algos:
            if opt_algo != 'tpe':
                runcount_dict[opt_algo] = list()
            else:
                count_list = list()
                for key in runcount_dict.keys():
                    count_list.append(np.mean(runcount_dict[key]))
                assert len(count_list) > 0
                tpe_runcount = int(np.min(count_list))
                print('='*50, tpe_runcount)

            result = dict()
            mode, eta = None, None
            # Parse the parameters for each optimizer.
            if opt_algo.startswith('mono_smbo'):
                mode = 2
                if len(opt_algo.split('_')) == 3:
                    _, _, mode = opt_algo.split('_')
                    mode = int(mode)
                    eta = 10
                    optimizer = 'mono_smbo'
            else:
                optimizer = opt_algo

            print('Test optimizer: %s' % optimizer)

            seeds = get_seeds(dataset, rep_num)
            for run_id in range(start_id, rep_num):
                if B > 0:
                    task_name = dataset + '_%s_%d_%d_%d' % (task_id, B, run_count, run_id)
                else:
                    task_name = dataset + '_%s_%d_%d' % (task_id, run_count, run_id)
                seed = seeds[run_id]

                runcount_const = run_count if opt_algo != 'tpe' else tpe_runcount
                # Construct the AutoML classifier.
                cls = Classifier(optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=runcount_const, runtime=B,
                    task_name=task_name, update_mode=mode, param=eta)

                # Load CASH intermediate infos.
                if optimizer == 'smbo':
                    file_id = 'smac'
                elif optimizer == 'tpe':
                    file_id = 'hyperopt'
                elif optimizer == 'mono_smbo':
                    file_id = 'mm_bandit_%d_smac' % mode
                else:
                    raise ValueError('Invalid optimizer!')

                tmp_task_id = '%s_%d' % (task_id, B) if B > 0 else task_id
                tmp_configs, tmp_perfs = load_infos(dataset, tmp_task_id, run_count, run_id, file_id)
                if opt_algo != 'tpe':
                    runcount_dict[opt_algo].append(len(tmp_configs))

                # Test the CASH performance on test set.
                cash_test_acc = cls.score(X_test, y_test)
                key_id = '%s_%d_%d_%s' % (dataset, run_count, run_id, optimizer)
                result[key_id] = [cash_test_acc]
                print(result)

                # Save the test result.
                with open('data/%s/%s_test_result_%s_%s_%d_%d_%d.pkl' %
                                  (dataset_id, dataset, opt_algo, task_id, run_count, rep_num, start_id), 'wb') as f:
                    pickle.dump(result, f)


if __name__ == "__main__":
    test_exp3_cost_aware()
