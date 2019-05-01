import sys
import argparse
import pickle
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


rep_num = args.rep
run_count = args.run_count
datasets = args.datasets.split(',')
algo_list = ['adaboost', 'random_forest', 'k_nearest_neighbors', 'gradient_boosting']
print(rep_num, run_count, datasets)


def test_claim():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from alphaml.utils.constants import MAX_INT

    perfs_list = list()
    for dataset in datasets:
        for run_id in range(rep_num):
            X, y, _ = load_data(dataset)
            dm = DataManager(X, y)
            seed = np.random.random_integers(MAX_INT)
            task_format = dataset + '_claim_%d'

            for optimizer in ['smbo']:
                cls = Classifier(include_models=algo_list, optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=run_count, task_name=task_format % run_id)
                print(cls.predict(X))

                file_id = 'data/%s_claim_%d_%s.data' % (dataset, run_id, 'smac')
                with open(file_id, 'rb') as f:
                    data = pickle.load(f)

                best_id = np.argmax(data['perfs'])
                best_value = data['perfs'][best_id]
                if data['perfs'].count(best_value) > 1:
                    stats = dict()
                    for conf, perf in zip(data['configs'], data['perfs']):
                        if perf == best_value:
                            est = conf['estimator']
                            if est not in stats:
                                stats[est] = 0
                            stats[est] += 1
                    tmp_id = np.argmax(stats.values())
                    best_estimator = list(stats.keys())[tmp_id]
                    print('=' * 20, best_value, stats)
                else:
                    best_estimator = data['configs'][best_id]['estimator']
                    print('='*20, data['perfs'][best_id], data['configs'][best_id])

                run_cnts = len([item for item in data['configs'] if item['estimator'] == best_estimator])

                task_format = dataset + '_claim_single_%d'
                cls = Classifier(include_models=[best_estimator], optimizer=optimizer, seed=seed).fit(
                    dm, metric='accuracy', runcount=run_cnts, task_name=task_format % run_id)
                print(cls.predict(X))

                file_id = 'data/%s_claim_single_%d_%s.data' % (dataset, run_id, 'smac')
                with open(file_id, 'rb') as f:
                    data_s = pickle.load(f)
                print('='*20 + 'single', max(data_s['perfs']))
                perfs_list.append((data['perfs'], data_s['perfs']))

    for item in perfs_list:
        item1, item2 = item
        print(len(item1), max(item1), len(item2), max(item2))
    print('='*50)
    print(perfs_list)


def plot(dataset, rep_num):
    perfs_list = list()
    for run_id in range(rep_num):
        file_id = 'data/%s/%s_claim_single_%d_%s.data' % (dataset, dataset, run_id, 'smac')
        with open(file_id, 'rb') as f:
            data = pickle.load(f)
        estimator = data['configs'][0]['estimator']
        perfs_1 = data['perfs']

        file_id = 'data/%s/%s_claim_%d_%s.data' % (dataset, dataset, run_id, 'smac')
        with open(file_id, 'rb') as f:
            data = pickle.load(f)
        perfs_2 = [item[1] for item in zip(data['configs'], data['perfs']) if item[0]['estimator'] == estimator]
        assert len(perfs_1) == len(perfs_2)
        perfs_list.append((perfs_1, perfs_2))

    cnt = 0
    for item in perfs_list:
        item1, item2 = item
        print(len(item1), max(item1), len(item2), max(item2))
        if max(item1) >= max(item2):
            cnt += 1

    print('='*50, cnt)


if __name__ == "__main__":
    # test_claim()
    plot('segment', 50)
