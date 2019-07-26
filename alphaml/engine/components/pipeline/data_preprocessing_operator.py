import typing
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, \
    MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, DATA_PERPROCESSING


class ImputeOperator(Operator):
    # TODO: Different inpute strategy
    def __init__(self, params=None, label_col=-1):
        super().__init__(DATA_PERPROCESSING, 'dp_imputer', params)
        self.label_col = label_col

    def operate(self, dm_list: typing.List):
        # The input of a ImputeOperator is a pd.Dataframe
        assert len(dm_list) == 1 and isinstance(dm_list[0], pd.DataFrame)

        input_df = dm_list[0]
        df = self.impute_df(input_df)
        dm = DataManager()

        # Set the feature types
        if dm.feature_types is None:
            dm.set_col_type(df, df.columns[self.label_col])
        data = df.values

        swap_data = data[:, -1]
        data[:, -1] = data[:, self.label_col]
        data[:, self.label_col] = swap_data
        dm.train_X = data[:, :-1]
        dm.train_y = data[:, -1]
        self.result_dm = dm

    def impute_categorical(self, col):
        return col.fillna(col.mode().iloc[0])

    def impute_float(self, col) -> pd.Series:
        mean_val = col.mean()
        return col.fillna(mean_val)

    def impute_discrete(self, col) -> pd.Series:
        mean_val = int(col.mean())
        return col.fillna(mean_val)

    def impute_col(self, col, datatype) -> pd.Series:
        if datatype == "categorical":
            return self.impute_categorical(col)
        elif datatype == "float":
            return self.impute_float(col)
        elif datatype == "discrete":
            return self.impute_discrete(col)
        else:
            raise TypeError("Require datatype to be categorical, float or discrete")

    def impute_df(self, df) -> pd.DataFrame:
        for col in list(df.columns):
            dtype = df[col].dtype
            if dtype in [np.int, np.int16, np.int32, np.int64]:
                df[col] = self.impute_col(df[col], "discrete")
            elif dtype in [np.float, np.float16, np.float32, np.float64, np.float128, np.double]:
                df[col] = self.impute_col(df[col], "float")
            elif dtype in [np.str, np.str_, np.string_, np.object]:
                df[col] = self.impute_col(df[col], "categorical")
            else:
                raise TypeError("Unknow data type:", dtype)
            print("Coltype:", dtype)
        return df


class LabelEncoderOperator(Operator):
    def __init__(self, params=None, label_col=-1):
        super().__init__(DATA_PERPROCESSING, 'dp_labelencoder', params)
        self.label_col = label_col
        self.label_encoder = LabelEncoder()

    def operate(self, dm_list: typing.List):
        # The input of a LabelEncoderOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        # Use LabelEncoder to transform categorical labels into numerical values
        dm.train_y = self.label_encoder.fit_transform(dm.train_y)
        self.result_dm = dm


class FeatureEncoderOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for OneHotEncoder, 1 for OrdinalEncoder
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_onehotencoder', params)
            self.encoder = OneHotEncoder(handle_unknown="ignore")
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_ordinalencoder', params)
            self.encoder = OrdinalEncoder()
        else:
            raise ValueError("Invalid params in FeatureEncoderOperator. Expected {0,1}")

    def operate(self, dm_list: typing.List):
        # The input of a FeatureEncoderOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types

        # Encode categorical features
        categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]
        other_index = [i for i in range(len(feature_types)) if feature_types[i] != "Categorical"]

        x = dm.train_X

        # Transform categorical features via *Encoder
        if self.params == 0:  # One-hot
            categorical_x = x[:, categorical_index]
            other_x = x[:, other_index]

            categorical_x = self.encoder.fit_transform(categorical_x).toarray()
            categorical_features = ["One-Hot"] * categorical_x.shape[1]
            other_features = [feature_types[i] for i in other_index]

            x = np.hstack((categorical_x, other_x)).astype(np.float)
            dm.feature_types = np.concatenate((categorical_features, other_features))
        elif self.params == 1:  # Ordinal
            x[:, categorical_index] = self.encoder.fit_transform(x[:, categorical_index])
            x = x.astype(np.float)

            for index in categorical_index:
                dm.feature_types[index] = 'Discrete'

        dm.train_X = x
        self.result_dm = dm


class ScalerOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for StandardScaler, 1 for MinMaxScaler, 2 for MaxAbsScaler
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_standardscaler', params)
            self.scaler = StandardScaler()
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_minmaxscaler', params)
            self.scaler = MinMaxScaler()
        elif params == 2:
            super().__init__(DATA_PERPROCESSING, 'dp_maxabsscaler', params)
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("Invalid params for ScalerOperator. Expected {0,1,2}")

    def operate(self, dm_list: typing.List):
        # The input of a ScalerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        train_x = dm.train_X
        scaler = self.scaler

        train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
        dm.train_X = train_x
        self.result_dm = dm


class NormalizerOperator(Operator):
    def __init__(self, params=0):
        '''
        :param params: 0 for l2 norm, 1 for l1 norm
        '''
        if params == 0:
            super().__init__(DATA_PERPROCESSING, 'dp_l2normalizer', params)
            self.normalizer = Normalizer(norm='l2')
        elif params == 1:
            super().__init__(DATA_PERPROCESSING, 'dp_l1normalizer', params)
            self.normalizer = Normalizer(norm='l1')
        else:
            raise ValueError("Invalid params for NormalizerOperator. Expected {0,1}")

    def operate(self, dm_list: typing.List):
        # The input of a NormalizerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]
        x = dm.train_X
        normalizer = self.normalizer

        x[:, numercial_index] = normalizer.fit_transform(x[:, numercial_index])
        dm.train_X = x
        self.result_dm = dm


# TODO: Bucketizer