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

    rep_num = 5
    run_count = 100
    datasets = ['australian_scale']
    for dataset in datasets:
        for run_id in range(rep_num):
            for optimizer in ['smbo', 'ts_smbo']:
                task_format = dataset + '_%d'
                X, y, _ = load_data(dataset)
                print(min(y), max(y))
                print(X.shape, y.shape)

                cls = Classifier(
                    include_models=['gaussian_nb', 'adaboost', 'random_forest', 'k_nearest_neighbors', 'gradient_boosting'],
                    optimizer=optimizer
                    ).fit(DataManager(X, y), metric='accuracy', runcount=run_count, task_name=task_format % run_id)
                print(cls.predict(X))


if __name__ == "__main__":
    test_cash_module()
