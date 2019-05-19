from alphaml.engine.components.feature_engineering.selector import *

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def test_ftest(x_train, x_test, y_train, y_test):
    selector = FtestSelector()
    selector.fit(x_train, y_train)
    x_test = selector.transform(x_test, 10)

    assert x_test.shape == (360, 10)


def test_mi(x_train, x_test, y_train, y_test):
    selector = MutualInformationSelector()
    selector.fit(x_train, y_train)
    x_test = selector.transform(x_test, 10)

    assert x_test.shape == (360, 10)


def test_chi2(x_train, x_test, y_train, y_test):
    selector = ChiSqSelector()
    selector.fit(x_train, y_train)
    x_test = selector.transform(x_test, 10)

    assert x_test.shape == (360, 10)


def test_rf(x_train, x_test, y_train, y_test):
    selector = RandomForestSelector()
    selector.fit(x_train, y_train)
    x_test = selector.transform(x_test, 10)

    assert x_test.shape == (360, 10)


def test_lasso(x_train, x_test, y_train, y_test):
    selector = LassoSelector()
    selector.fit(x_train, y_train)
    x_test = selector.transform(x_test, 10)

    assert x_test.shape == (360, 10)


if __name__ == '__main__':
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

    test_ftest(x_train, x_test, y_train, y_test)
    test_mi(x_train, x_test, y_train, y_test)
    test_chi2(x_train, x_test, y_train, y_test)
    test_rf(x_train, x_test, y_train, y_test)
    test_lasso(x_train, x_test, y_train, y_test)
