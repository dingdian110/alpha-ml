import os
import sys
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='master')
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--run_count', type=int, default=5000)
parser.add_argument('--B', type=int, default=30)
parser.add_argument('--datasets', type=str, default='pc4')
parser.add_argument('--mth', type=str, default='1,2')
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


def test_exp6_evaluation():
    rep_num = args.rep
    run_count = args.run_count

    start_id = args.start_runid
    datasets = args.datasets.split(',')
    algo_ids = [int(id) for id in args.mth.split(',')]
    print(rep_num, run_count, datasets)
    task_id = "exp_6_evaluation"

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

        optimizer_algos = ['smbo', 'tpe', 'cmab_ts', 'mono_smbo_3_0']
        optimizer_algos = [optimizer_algos[id] for id in algo_ids]
        # optimizer_algos = ['smbo']
        # Test each optimizer algorithm:
        for opt_algo in optimizer_algos:
            result = dict()
            mode, eta = None, None
            # Parse the parameters for each optimizer.
            if opt_algo.startswith('mono_smbo'):
                optimizer = 'mono_smbo'
                mode, eta = 3, 10
            elif opt_algo.startswith('baseline'):
                optimizer = 'baseline'
                mode = 2
            elif opt_algo.startswith('rl'):
                if len(opt_algo.split('_')) == 3:
                    _, mode, eta = opt_algo.split('_')
                    mode = int(mode)
                    optimizer = 'rl_smbo'
                    eta = float(eta)
            else:
                optimizer = opt_algo

            print('Test optimizer: %s' % optimizer)

            seeds = get_seeds(dataset, rep_num)
            for run_id in range(start_id, rep_num):
                task_name = dataset + '_%s_%d_%d' % (task_id, run_count, run_id)
                seed = seeds[run_id]

                # Construct the AutoML classifier.
                cls = Classifier(optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=run_count, runtime=None,
                    task_name=task_name, update_mode=mode, param=eta)

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
    test_exp6_evaluation()
