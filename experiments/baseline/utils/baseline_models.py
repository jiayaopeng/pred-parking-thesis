from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def logistic_regression(X, y, X_test):
    lr = LogisticRegression(random_state=0, class_weight="balanced").fit(X, y)
    preds = lr.predict(X_test)

    return lr, preds  # return the trained model object and a list of predictions


def random_forest(X, y, X_test):
    rf = RandomForestClassifier(
        max_depth=2, random_state=42, class_weight="balanced"
    ).fit(X, y)
    preds = rf.predict(X_test)

    return rf, preds


def catboost(X, y, cat_feat, X_test, iterations=1000):
    cb = CatBoostClassifier(iterations=iterations, auto_class_weights="Balanced").fit(
        X, y, cat_features=cat_feat, metric_period=200
    )
    preds = cb.predict(X_test)

    return cb, preds
