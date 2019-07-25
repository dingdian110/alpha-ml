
def test_impute():
    print("Test Imputer")
    import pandas as pd
    from alphaml.engine.components.pipeline.data_preprocessing_operator import ImputeOperator
    imop=ImputeOperator()
    input_pd=pd.read_csv("data/cls_data/eucalyptus/eucalyptus.csv",na_values=['?'])
    imop.operate([input_pd])
    print(imop.result_dm.train_X)
    print(imop.result_dm.train_y)
    return imop.result_dm

def test_labelencoder():
    print("Test Label Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import LabelEncoderOperator
    dm=test_impute()
    leop=LabelEncoderOperator()
    leop.operate([dm])
    print(leop.result_dm.train_y)
    return leop.result_dm

def test_onehotencoder():
    print("Test One Hot Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import OneHotEncoderOperator
    dm=test_labelencoder()
    ohop=OneHotEncoderOperator()
    ohop.operate([dm])
    print(ohop.result_dm.train_X)
    print("One-hot dimension:",ohop.result_dm.train_X.shape[1])
    return ohop.result_dm

def test_ordinalencoder():
    print("Test Ordinal Encoder")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import OrdinalEncoderOperator
    dm = test_labelencoder()
    ohop = OrdinalEncoderOperator()
    ohop.operate([dm])
    print(ohop.result_dm.train_X)
    print("Ordinal dimension:", ohop.result_dm.train_X.shape[1])
    return ohop.result_dm

def test_minmaxscaler():
    print("Test MinMaxScaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import MinMaxScalerOperator
    dm = test_onehotencoder()
    print([dm.feature_types])
    mmop = MinMaxScalerOperator()
    mmop.operate([dm])
    print(mmop.result_dm.train_X)
    return mmop.result_dm

def test_standardscaler():
    print("Test StandardScaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import StandardScalerOperator
    dm = test_onehotencoder()
    print([dm.feature_types])
    sdop = StandardScalerOperator()
    sdop.operate([dm])
    print(sdop.result_dm.train_X)
    return sdop.result_dm

def test_maxabsscaler():
    print("Test MaxAbsScaler")
    from alphaml.engine.components.pipeline.data_preprocessing_operator import MaxAbsScalerOperator
    dm = test_onehotencoder()
    print([dm.feature_types])
    maop = MaxAbsScalerOperator()
    maop.operate([dm])
    print(maop.result_dm.train_X)
    return maop.result_dm