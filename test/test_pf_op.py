def test_impute():
    print("Test Imputer")
    import pandas as pd
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputeOperator
    imop = ImputeOperator()
    input_pd = pd.read_csv("data/cls_data/eucalyptus/eucalyptus.csv", na_values=['?'])
    imop.operate([input_pd])
    print(imop.result_dm.train_X)
    print(imop.result_dm.train_y)
    return imop.result_dm


def test_labelencoder():
    print("Test Label Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import LabelEncoderOperator
    dm = test_impute()
    leop = LabelEncoderOperator()
    leop.operate([dm])
    print(leop.result_dm.train_y)
    return leop.result_dm


def test_featureencoder():
    print("Test Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import FeatureEncoderOperator
    dm = test_labelencoder()
    ohop = FeatureEncoderOperator(0)
    ohop.operate([dm])
    print(ohop.result_dm.train_X)
    print("One-hot dimension:", ohop.result_dm.train_X.shape[1])
    return ohop.result_dm


def test_scaler():
    print("Test Scaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ScalerOperator
    dm = test_featureencoder()
    print([dm.feature_types])
    mmop = ScalerOperator(params=0)
    mmop.operate([dm])
    print(mmop.result_dm.train_X)
    return mmop.result_dm


def test_normalizer():
    print("Test NormalizerScaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import NormalizerOperator
    dm = test_featureencoder()
    print([dm.feature_types])
    mmop = NormalizerOperator(params=0)
    mmop.operate([dm])
    print(mmop.result_dm.train_X)
    return mmop.result_dm


def test_pf():
    print("Test Polynomial")
    from alphaml.engine.components.pipeline.feature_generation_operator import PolynomialFeaturesOperator
    dm = test_featureencoder()
    print([dm.feature_types])
    mmop = PolynomialFeaturesOperator(2)
    mmop.operate([dm])
    print("New feature dimension:", mmop.result_dm.train_X.shape[1])
    print(mmop.result_dm.train_X)
    return mmop.result_dm


def test_kbest():
    print("Test K-best")
    from alphaml.engine.components.pipeline.feature_selection_operator import NaiveSelectorOperator
    dm = test_featureencoder()
    print([dm.feature_types])
    mmop = NaiveSelectorOperator([50, 0])
    mmop.operate([dm, dm])
    print("Number of samples:", dm.train_X.shape[0], '->', mmop.result_dm.train_X.shape[0])
    print("New feature dimension:", dm.train_X.shape[1], '->', mmop.result_dm.train_X.shape[1])
    print(mmop.result_dm.train_X)
    return mmop.result_dm


def test_ml():
    print("Test ML")
    from alphaml.engine.components.pipeline.feature_selection_operator import MLSelectorOperator
    dm = test_featureencoder()
    print([dm.feature_types])
    mmop = MLSelectorOperator([50, 0, 1])
    mmop.operate([dm, dm])
    print("Number of samples:", dm.train_X.shape[0], '->', mmop.result_dm.train_X.shape[0])
    print("New feature dimension:", dm.train_X.shape[1], '->', mmop.result_dm.train_X.shape[1])
    print(mmop.result_dm.train_X)
    return mmop.result_dm
