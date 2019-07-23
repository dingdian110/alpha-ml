from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def metrics_func(metrics, x_train, y_train, x_valid, y_valid, model):
    if metrics == "accuracy":
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        return accuracy_score(y_valid, y_pred)
    elif metrics == "auc":
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_valid)[:, 1]
        return roc_auc_score(y_valid, y_pred)
    elif metrics == "mse":
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        return mean_squared_error(y_valid, y_pred)
    else:
        raise ValueError()
