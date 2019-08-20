import os
import sys
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213', 'gc'], default='master')
parser.add_argument('--start_runid', type=int, default=0)
parser.add_argument('--rep', type=int, default=50)
parser.add_argument('--run_count', type=int, default=500)
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
from alphaml.utils.rand import get_seeds


def evaluate_k():
    algo_list = ['xgboost', 'liblinear_svc', 'gradient_boosting', 'decision_tree', 'passive_aggressive', 'qda',
                 'random_forest', 'sgd', 'extra_trees', 'lda', 'gaussian_nb', 'libsvm_svc',
                 'logistic_regression', 'adaboost', 'k_nearest_neighbors']

    rep_num = args.rep
    run_count = args.run_count
    start_id = args.start_runid
    datasets = args.datasets.split(',')
    task_id = 'exp5_eval_k'
    print(rep_num, run_count, datasets, task_id)

    for dataset in datasets:
        # Make directories.
        dataset_id = dataset.split('_')[0]
        save_dir = "data/%s/" % dataset_id
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dataset partition.
        X, y, _ = load_data(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        dm = DataManager(X_train, y_train)

        # opt_algos = ['mono_smbo_3_0', 'smbo', 'baseline_2', 'tpe']
        opt_algos = ['mono_smbo_3_0', 'smbo', 'baseline_2']
        for algo in opt_algos:
            result = dict()
            seeds = get_seeds(dataset, rep_num)
            for run_id in range(start_id, rep_num):
                seed = seeds[run_id]

                # Test each optimizer algorithm:
                for n_est in [1, 2, 4, 8, 12, 15]:
                    algos = algo_list[:n_est]
                    task_name = dataset + '_%s_%d_%d_%d' % (task_id, run_count, run_id, n_est)
                    mode, param = 3, None
                    if algo.startswith('mono_smbo'):
                        optimizer = 'mono_smbo'
                        mode, param = 3, 10
                    elif algo.startswith('baseline'):
                        optimizer = 'baseline'
                        mode = 2
                    else:
                        optimizer = algo

                    print('Test %s optimizer => %s' % (optimizer, task_name))

                    # Construct the AutoML classifier.
                    cls = Classifier(optimizer=optimizer, seed=seed, include_models=algos).fit(
                        dm, metric='accuracy', runcount=run_count, task_name=task_name, update_mode=mode, param=param)
                    acc = cls.score(X_test, y_test)
                    key_id = '%s_%d_%d_%d_%s' % (dataset, run_count, n_est, run_id, optimizer)
                    result[key_id] = acc

                # Display and save the test result.
                print(result)

                with open('data/%s/%s_test_%s_%d_%d_%d.pkl' % (dataset_id, dataset, algo, run_count, rep_num,
                                                               start_id), 'wb') as f:
                    pickle.dump(result, f)


if __name__ == "__main__":
    evaluate_k()
