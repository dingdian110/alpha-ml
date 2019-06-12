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


def test_auto():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.imgclassifier import ImageClassifier
    from alphaml.datasets.img_cls_dataset import load_data_img, load_data
    import pickle
    import os
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    traindir, validdir = load_data_img("cifar10")
    train_x, train_y, _, _, _ = load_data('cifar10')
    # ImageClassifier(include_models=['resnet'], optimizer='smbo').fit_from_directory([traindir, validdir])
    # ImageClassifier(include_models=['resnet'], optimizer='smbo').fit(DataManager(train_x, train_y), metric='acc')
    ImageClassifier(include_models=['resnet'], optimizer='smbo').fit_from_directory(traindir, metric='acc')


if __name__ == "__main__":
    test_auto()
