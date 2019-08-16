import pandas as pd


def test_impute():
    print("Test Imputer")
    import pandas as pd
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator
    imoptr = ImputerOperator()
    input_pd = pd.read_csv("data/cls_data/eucalyptus/eucalyptus.csv", na_values=['?'])
    trdm = imoptr.operate([input_pd], 'train')
    print(trdm.train_X)
    # print(trdm.train_y)
    tsdm = imoptr.operate([input_pd.drop(input_pd.columns[len(input_pd.columns) - 1], axis=1, inplace=False)], 'test')
    print(tsdm.test_X)
    return trdm, tsdm


def test_labelencoder():
    print("Test Label Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import LabelEncoderOperator
    trdm, tsdm = test_impute()
    leop = LabelEncoderOperator()
    trdm = leop.operate([trdm], phase='train')
    print(trdm.train_y)
    tsdm = leop.operate([tsdm], phase='test')
    print(tsdm.train_y)
    return trdm, tsdm


def test_featureencoder():
    print("Test Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import FeatureEncoderOperator
    trdm, tsdm = test_labelencoder()
    ohop = FeatureEncoderOperator(params=0)
    trdm = ohop.operate([trdm], phase='train')
    print(trdm.train_X)
    print("One-hot dimension:", trdm.train_X.shape[1])
    tsdm = ohop.operate([tsdm], phase='test')
    print(tsdm.test_X)
    print("One-hot dimension:", tsdm.test_X.shape[1])
    return trdm, tsdm


def test_scaler():
    print("Test Scaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ScalerOperator
    trdm, tsdm = test_featureencoder()
    mmop = ScalerOperator(params=0)
    trdm = mmop.operate([trdm], phase='train')
    print(trdm.train_X)
    tsdm = mmop.operate([tsdm], phase='test')
    print(tsdm.test_X)
    return trdm, tsdm


def test_normalizer():
    print("Test NormalizerScaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import NormalizerOperator
    trdm, tsdm = test_featureencoder()
    print([trdm.feature_types])
    mmop = NormalizerOperator(params=0)
    trdm = mmop.operate([trdm], phase='train')
    print(trdm.train_X)
    tsdm = mmop.operate([tsdm], phase='test')
    print(tsdm.test_X)

    return trdm, tsdm


def test_pf():
    print("Test Polynomial")
    from alphaml.engine.components.pipeline.feature_generation_operator import PolynomialFeaturesOperator
    trdm, tsdm = test_featureencoder()
    mmop = PolynomialFeaturesOperator(params=2)
    trdm = mmop.operate([trdm], phase='train')
    print("New feature dimension:", trdm.train_X.shape[1])
    print(trdm.train_X)
    tsdm = mmop.operate([tsdm], phase='test')
    print("New feature dimension:", tsdm.test_X.shape[1])
    print(tsdm.test_X)
    return trdm, tsdm


def test_pca():
    print("Test PCA")
    from alphaml.engine.components.pipeline.feature_generation_operator import PCAOperator
    trdm, tsdm = test_featureencoder()
    mmop = PCAOperator(params=5)
    trdm = mmop.operate([trdm], phase='train')
    print("New feature dimension:", trdm.train_X.shape[1])
    print(trdm.train_X)
    tsdm = mmop.operate([tsdm], phase='test')
    print("New feature dimension:", tsdm.test_X.shape[1])
    print(tsdm.test_X)
    return trdm, tsdm


def test_autocross():
    print("Test AutoCross")
    from alphaml.engine.components.pipeline.feature_generation_operator import AutoCrossOperator
    trdm, tsdm = test_featureencoder()
    acop = AutoCrossOperator(stratify=True, metric='acc', params=50)
    trdm = acop.operate([trdm], phase='train')
    print("New feature dimension:", trdm.train_X.shape[1])
    print(trdm.train_X)
    tsdm = acop.operate([tsdm], phase='test')
    print("New feature dimension:", tsdm.test_X.shape[1])
    print(tsdm.test_X)
    return trdm, tsdm


def test_kbest():
    print("Test K-best")
    from alphaml.engine.components.pipeline.feature_selection_operator import NaiveSelectorOperator
    trdm, tsdm = test_featureencoder()
    print([trdm.feature_types])
    mmop = NaiveSelectorOperator(params=[50, 1])
    newtrdm = mmop.operate([trdm, trdm])
    print("Number of samples:", trdm.train_X.shape[0], '->', newtrdm.train_X.shape[0])
    print("New feature dimension:", trdm.train_X.shape[1], '->', newtrdm.train_X.shape[1])
    print(newtrdm.train_X)
    newtsdm = mmop.operate([tsdm, tsdm], phase='test')
    print("Number of samples:", tsdm.test_X.shape[0], '->', newtsdm.test_X.shape[0])
    print("New feature dimension:", tsdm.test_X.shape[1], '->', newtsdm.test_X.shape[1])
    print(newtsdm.test_X)
    return newtrdm, newtsdm


def test_ml():
    print("Test ML")
    from alphaml.engine.components.pipeline.feature_selection_operator import MLSelectorOperator
    trdm, tsdm = test_featureencoder()
    print([trdm.feature_types])
    mmop = MLSelectorOperator(params=[50, 0, 0])
    newtrdm = mmop.operate([trdm, trdm])
    print("Number of samples:", trdm.train_X.shape[0], '->', newtrdm.train_X.shape[0])
    print("New feature dimension:", trdm.train_X.shape[1], '->', newtrdm.train_X.shape[1])
    print(newtrdm.train_X)
    newtsdm = mmop.operate([tsdm, tsdm], phase='test')
    print("Number of samples:", tsdm.test_X.shape[0], '->', newtsdm.test_X.shape[0])
    print("New feature dimension:", tsdm.test_X.shape[1], '->', newtsdm.test_X.shape[1])
    print(newtsdm.test_X)
    return newtrdm, newtsdm


def test_remove_const():
    print("Test RC")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator, ConstantRemoverOperator
    df = pd.DataFrame({'f1': [0, 0, 2], 'f2': [1, 1, 1], 'f3': [1, 2, 1], 'f4': [2, 2, 2], 'l': [3, 0, 3]})
    op1 = ImputerOperator()
    op2 = ConstantRemoverOperator()
    dm = op1.operate([df])
    dm = op2.operate([dm])
    print(dm.train_X)
    df = df.drop(columns='l')
    dm = op1.operate([df], phase='test')
    dm = op2.operate([dm], phase='test')
    print(dm.test_X)


def test_remove_identical():
    print("Test RI")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator, IdenticalRemoverOperator
    df = pd.DataFrame({'f1': [0, 0, 2], 'f2': [1, 2, 2], 'f3': [1, 2, 2], 'f4': [1, 2, 2], 'l': [3, 0, 3]})
    op1 = ImputerOperator()
    op2 = IdenticalRemoverOperator()
    dm = op1.operate([df])
    dm = op2.operate([dm])
    print(dm.train_X)
    df = df.drop(columns='l')
    dm = op1.operate([df], phase='test')
    dm = op2.operate([dm], phase='test')
    print(dm.test_X)

def test_remove_variance():
    print("Test RV")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator, VarianceRemoverOperator
    df = pd.DataFrame({'f1': [0, 0, 2], 'f2': [1, 1, 1.0001], 'f3': [1, 2, 1], 'f4': [2, 2, 2], 'l': [3, 0, 3]})
    op1 = ImputerOperator()
    op2 = VarianceRemoverOperator(params=1e-5)
    dm = op1.operate([df])
    dm = op2.operate([dm])
    print(dm.train_X)
    df = df.drop(columns='l')
    dm = op1.operate([df], phase='test')
    dm = op2.operate([dm], phase='test')
    print(dm.test_X)


def test_zero():
    print("Test zero")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputerOperator
    from alphaml.engine.components.pipeline.feature_generation_operator import ZeroOperator
    df = pd.DataFrame({'f1': [0, 0, 2], 'f2': [1, 1, 1.0001], 'f3': [1, 2, 1], 'f4': [2, 0, 0], 'l': [3, 0, 3]})
    op1 = ImputerOperator()
    op2 = ZeroOperator()
    dm = op1.operate([df])
    dm = op2.operate([dm])
    print(dm.train_X)
    df = df.drop(columns='l')
    dm = op1.operate([df], phase='test')
    dm = op2.operate([dm], phase='test')
    print(dm.test_X)
