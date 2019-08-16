import os
import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213'], default='master')
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--datasets', type=str, default='glass')
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
    rep_num = 20
    run_count = 200
    start_id = args.start_runid
    datasets = args.datasets.split(',')
    task_id = "eval_non_tmp"
    print(rep_num, run_count, datasets, task_id)

    result = dict()
    for dataset in datasets:
        dataset_id = dataset.split('_')[0]
        result_dir = 'data/'+dataset_id
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        seeds = get_seeds(dataset, rep_num)
        for run_id in range(start_id, rep_num):
            task_name = dataset + '_%s_%d_%d' % (task_id, run_count, run_id)
            seed = seeds[run_id]

            # Dataset partition.
            X, y, _ = load_data(dataset)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            dm = DataManager(X_train, y_train)

            optimizer_algos = ['rl_2_1', 'rl_3_0']
            # Test each optimizer algorithm:
            for optimizer in optimizer_algos:
                mode, eta = None, None
                # Parse the parameters for each optimizer.
                if optimizer.startswith('rl'):
                    if len(optimizer.split('_')) == 3:
                        _, mode, eta = optimizer.split('_')
                        mode = int(mode)
                        optimizer = 'rl_smbo'
                        eta = float(eta)
                    else:
                        raise ValueError('Wrong SH params!')

                if optimizer.startswith('ucb_smbo'):
                    mode = 1
                    if len(optimizer.split('_')) == 3:
                        _, _, mode = optimizer.split('_')
                        mode = int(mode)
                        optimizer = 'ucb_smbo'

                print('Test %s optimizer => %s' % (optimizer, task_name))

                # Construct the AutoML classifier.
                cls = Classifier(optimizer=optimizer, seed=seed, include_models=['xgboost']).fit(
                    dm, metric='accuracy', runcount=run_count,
                    task_name=task_name, update_mode=mode, param=eta)
                acc = cls.score(X_test, y_test)
                key_id = '%s_%d_%d_%s' % (dataset, run_count, run_id, optimizer)
                result[key_id] = acc

            # Display and save the test result.
            print(result)
            with open('data/%s/%s_test_result_%s_%d_%d_%d.pkl' %
                              (dataset_id, dataset_id, task_id, run_count, rep_num, start_id), 'wb') as f:
                pickle.dump(result, f)


if __name__ == "__main__":
    test_cash_module()
