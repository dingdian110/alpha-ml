import typing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, \
    MinMaxScaler, StandardScaler, MaxAbsScaler
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


class OneHotEncoderOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_onehotencoder', params)
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore")

    def operate(self, dm_list: typing.List):
        # The input of a OneHotEncoderOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]
        other_index = [i for i in range(len(feature_types)) if feature_types[i] != "Categorical"]

        x = dm.train_X
        categorical_x = x[:, categorical_index]
        other_x = x[:, other_index]

        # Transform categorical features via OneHotEncoder
        self.onehot_encoder.fit(categorical_x)
        categorical_x = self.onehot_encoder.transform(categorical_x).toarray()

        categorical_features = ["One-Hot"] * categorical_x.shape[1]
        other_features = [feature_types[i] for i in other_index]

        x = np.hstack((categorical_x, other_x)).astype(np.float)
        dm.feature_types = np.concatenate((categorical_features, other_features))

        dm.train_X = x
        self.result_dm = dm


class OrdinalEncoderOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_oridinalencoder', params)
        self.oridinal_encoder = OrdinalEncoder()

    def operate(self, dm_list: typing.List):
        # The input of a OrdinalOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        categorical_index = [i for i in range(len(feature_types)) if feature_types[i] == "Categorical"]

        # Transform categorical features via OrdinalEncoder
        x = dm.train_X
        x[:, categorical_index] = self.oridinal_encoder.fit_transform(x[:, categorical_index])
        x = x.astype(np.float)

        for index in categorical_index:
            dm.feature_types[index] = 'Discrete'

        dm.train_X = x
        self.result_dm = dm


# TODO: Bucketizer

class MinMaxScalerOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_minmaxscaler', params)
        self.minmaxscaler = MinMaxScaler()

    def operate(self, dm_list: typing.List):
        # The input of a MinMaxScalerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        train_x = dm.train_X
        scaler = self.minmaxscaler

        train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
        dm.train_X = train_x
        self.result_dm = dm


class StandardScalerOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_standardscaler', params)
        self.standardscaler = StandardScaler()

    def operate(self, dm_list: typing.List):
        # The input of a StandardScalerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        train_x = dm.train_X
        scaler = self.standardscaler

        train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
        dm.train_X = train_x
        self.result_dm = dm


class MaxAbsScalerOperator(Operator):
    def __init__(self, params=None):
        super().__init__(DATA_PERPROCESSING, 'dp_maxabsscaler', params)
        self.maxabsscaler = MaxAbsScaler()

    def operate(self, dm_list: typing.List):
        # The input of a MaxAbsScalerOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]

        train_x = dm.train_X
        scaler = self.maxabsscaler

        train_x[:, numercial_index] = scaler.fit_transform(train_x[:, numercial_index])
        dm.train_X = train_x
        self.result_dm = dm