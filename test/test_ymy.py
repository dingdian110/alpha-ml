import warnings
from sklearn.preprocessing import LabelEncoder
import sys
warnings.filterwarnings("ignore")
sys.path.append("/home/daim_gpu/sy/AlphaML")

def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from alphaml.datasets.cls_dataset.dataset_loader import load_data
    from sklearn.model_selection import train_test_split
    import random
    from sklearn.metrics import roc_auc_score
    result = []
    for i in range(10):
        import xlrd
        sheet = xlrd.open_workbook("lyqdata.xlsx")
        sheet = sheet.sheet_by_index(0)
        nrows = sheet.nrows
        X_train = []
        y_train = []
        for i in range(2, nrows):
            X_train.append(sheet.row_values(i, start_colx=1))
            y_train.append(int(sheet.cell_value(i, 0)))

        dm = DataManager(X_train, y_train, val_size=0.33, random_state=random.randint(1, 255))
        cls = Classifier(#exclude_models=['xgboost', 'gradient_boosting'],
                         optimizer='mono_smbo',
                         ensemble_method='ensemble_selection',
                         ensemble_size=50,
                         ).fit(dm, metric='auc', runcount=250)

        sheet = xlrd.open_workbook("lyqtestdata.xlsx")
        sheet = sheet.sheet_by_index(0)
        nrows = sheet.nrows
        X_test = []
        y_test = []
        for i in range(1, nrows):
            X_test.append(sheet.row_values(i, start_colx=1))
            y_test.append(int(sheet.cell_value(i, 0)))
        pred = cls.predict(X_test)
        result.append(roc_auc_score(y_test, pred))
        print(result)

    import pickle
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    test_cash_module()
