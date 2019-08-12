import os
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213'], default='master')
parser.add_argument('--rep', type=int, default=10)
parser.add_argument('--run_count', type=int, default=200)
parser.add_argument('--datasets', type=str, default='iris')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def test_estimator():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from alphaml.utils.constants import MAX_INT

    rep_num = args.rep
    run_count = args.run_count
    datasets = args.datasets.split(',')
    print(rep_num, run_count, datasets)

    for dataset in datasets:
        dataset_id = dataset.split('_')[0]
        result_dir = 'data/' + dataset_id
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        task_format = dataset + '_est_%d'
        X, y, _ = load_data(dataset)
        dm = DataManager(X, y)
        seed = np.random.random_integers(MAX_INT)
        for optimizer in ['smbo']:
            cls = Classifier(
                include_models=['gradient_boosting'],
                optimizer=optimizer,
                seed=seed
            ).fit(
                dm, metric='accuracy', runcount=run_count, task_name=task_format)
            print(cls.predict(X))


if __name__ == "__main__":
    test_estimator()
