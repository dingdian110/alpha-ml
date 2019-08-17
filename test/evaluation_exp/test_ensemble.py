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
from alphaml.engine.components.ensemble.ensemble_selection import EnsembleSelection
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score


def get_seeds(dataset, rep_num):
    # Map the dataset to a fixed integer.
    dataset_id = int(''.join([str(ord(c)) for c in dataset[:6] if c.isalpha()])) % 100000
    np.random.seed(dataset_id)
    return np.random.random_integers(10000, size=rep_num)


def load_infos(dataset, task_id, run_count, id, mth):
    data_folder = '/home/thomas/PycharmProjects/alpha-ml/data/'
    tmp_d = dataset.split('_')[0]
    file_id = data_folder + '%s/%s_%s_%d_%d_%s.data' % (tmp_d, dataset, task_id, run_count, id, mth)
    with open(file_id, 'rb') as f:
        data = pickle.load(f)
    configs, perfs = data['configs'], data['perfs']
    assert len(configs) == len(perfs)
    return configs, perfs


def test_ensemble():
    rep_num = 3
    run_count = 100
    start_id = args.start_runid
    datasets = args.datasets.split(',')
    task_id = "eval_ensemble"
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

            optimizer_algos = ['mono_smbo_3_0']
            # Test each optimizer algorithm:
            for optimizer in optimizer_algos:
                mode, eta = None, None
                # Parse the parameters for each optimizer.
                if optimizer.startswith('mono_smbo'):
                    mode = 2
                    if len(optimizer.split('_')) == 4:
                        _, _, mode, r = optimizer.split('_')
                        mode, r = int(mode), int(r)
                        eta = 10
                        optimizer = 'mono_smbo'

                print('Test %s optimizer => %s' % (optimizer, task_name))

                # Construct the AutoML classifier.
                cls = Classifier(optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=run_count,
                    task_name=task_name, update_mode=mode, param=eta)

                # Test the CASH performance on test set.
                cash_score = cls.score(X_test, y_test)

                # Load CASH intermediate infos.
                tmp_configs, tmp_perfs = load_infos(dataset, task_id, run_count, run_id, 'mm_bandit_3_smac')
                model_infos = (tmp_configs, tmp_perfs)
                ensemble_size = 10
                task_type = type_of_target(dm.train_y)
                metric = accuracy_score

                ensemble_model = EnsembleSelection(model_infos, ensemble_size, task_type, metric, n_best=7)
                ensemble_model.fit(dm)

                ens_pred_val = ensemble_model.predict(dm.val_X)
                ens_score_val = accuracy_score(ens_pred_val, dm.val_y)

                ens_pred = ensemble_model.predict(X_test)
                ens_score = accuracy_score(ens_pred, y_test)

                key_id = '%s_%d_%d_%s' % (dataset, run_count, run_id, optimizer)
                result[key_id] = [cash_score, ens_score_val, ens_score]
                print(result)

            # Display and save the test result.
            print(result)
            with open('data/%s/%s_test_result_%s_%d_%d_%d.pkl' %
                              (dataset_id, dataset_id, task_id, run_count, rep_num, start_id), 'wb') as f:
                pickle.dump(result, f)


if __name__ == "__main__":
    test_ensemble()
