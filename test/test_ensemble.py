import sklearn
from sklearn.datasets import load_iris,load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from alphaml.engine.components.ensemble.Blending import Blending

def test_Blending():
    data = load_digits()
    X, y = data.data, data.target
    # Make train/test split
    # As usual in machine learning task we have X_train, y_train, and X_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=0, stratify=y_train)
    models = [
        ExtraTreesClassifier(random_state=0, n_jobs=-1,
                             n_estimators=100, max_depth=3),

        RandomForestClassifier(random_state=0, n_jobs=-1,
                               n_estimators=100, max_depth=3),

        XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                      n_estimators=100, max_depth=3)
    ]

    Meta_learner = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                                 n_estimators=100, max_depth=3)

    fitted_models = []

    for model in models:
        fitted_model = model.fit(X_train, y_train)
        y_pre = fitted_model.predict(X_test)
        print(model.__class__.__name__)
        print('prediction score: [%.8f]' % accuracy_score(y_test, y_pre))
        fitted_models.append(fitted_model)
    #
    '''Using Blending'''
    ens = Blending(regression=False, needs_proba=True, metric=None, models=models, meta_learner=Meta_learner,
                   verbose=True)
    ens.fit(X_val, y_val)
    y_pre = ens.predict(X_test)
    print(y_pre)
    print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pre))

