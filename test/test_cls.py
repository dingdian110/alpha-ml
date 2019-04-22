import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['master', 'daim213'], default='master')
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--run_count', type=int, default=100)
parser.add_argument('--datasets', type=str, default='iris')
args = parser.parse_args()

if args.mode == 'master':
    sys.path.append('/home/thomas/PycharmProjects/alpha-ml')
elif args.mode == 'daim213':
    sys.path.append('/home/liyang/codes/alpha-ml')
else:
    raise ValueError('Invalid mode: %s' % args.mode)


def test_configspace():
    from alphaml.engine.components.components_manager import ComponentsManager
    from alphaml.engine.components.models.classification import _classifiers

    # print(_classifiers)
    # for item in _classifiers:
    #     name, cls = item, _classifiers[item]
    #     print(cls.get_hyperparameter_search_space())
    cs = ComponentsManager().get_hyperparameter_search_space(3)
    print(cs.sample_configuration(5))
    # print(cs.get_default_configuration())


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data

    rep_num = args.rep
    run_count = args.run_count
    datasets = args.datasets.split(',')
    print(rep_num, run_count, datasets)

    for dataset in datasets:
        for run_id in range(rep_num):
            for optimizer in ['smbo', 'ts_smbo']:
                task_format = dataset + '_%d'
                X, y, _ = load_data(dataset)

                cls = Classifier(
                    include_models=['gaussian_nb', 'adaboost', 'random_forest', 'k_nearest_neighbors', 'gradient_boosting'],
                    optimizer=optimizer
                    ).fit(DataManager(X, y), metric='accuracy', runcount=run_count, task_name=task_format % run_id)
                print(cls.predict(X))


if __name__ == "__main__":
    test_cash_module()
