import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    fbeta_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

import baseline_models as bm


def select_data(col: list, data: pd.DataFrame):
    """
    input data and feature columns, output the X, auxiliary columns(study_area, geometry, observation interval) and y column
    """
    sup_cols = ["street_id", "observation_interval_start"]
    aux_cols = ["study_area", "geometry"]
    X = data.loc[:, col].set_index(sup_cols).drop(["geometry", "availability"], axis=1)
    y_columns = ["availability"] + sup_cols
    y = data.loc[:, y_columns].set_index(sup_cols)
    aux_columns = aux_cols + sup_cols
    aux = data.loc[:, aux_columns].set_index(sup_cols)

    return X, y, aux


def normalize_data(X_train, X_valid, X_test):
    """
    Input splitted data from train, test, and valid, and output the standardized data
    """
    # we need to firstly split the data and then use the the scaling fitted on train to scale test and val
    train_scaler = StandardScaler()
    X_train_scaled = train_scaler.fit_transform(X_train)
    X_valid_scaled = train_scaler.transform(X_valid)
    X_test_scaled = train_scaler.transform(X_test)

    return X_train_scaled, X_valid_scaled, X_test_scaled


def create_train_val_test(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """create train, validation and test data"""
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=42
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_source_target_data(area_split, data):
    """ """
    target_data_ = data[data["study_area"].isin(area_split["Target"])]
    source_data_ = data[data["study_area"].isin(area_split["Source"])]

    target_data = target_data_.drop(["study_area"], axis=1)
    source_data = source_data_.drop(["study_area"], axis=1)

    return source_data, target_data


def evaluate_baseline(y_pred, y_true, model_name):
    """
    Input:
        y_pred: a list of prediction from the model
        y_true: a list of true values
        model_name: name of the model used, later will be stored as key
    Output:
        model result :model name as key, result dictionary including multiple metrics as value
    """
    model_result = {}
    result = {}

    result["AUC"] = roc_auc_score(y_score=y_pred, y_true=y_true)
    result["Recall"] = recall_score(y_true=y_true, y_pred=y_pred)
    result["Precision"] = precision_score(y_true=y_true, y_pred=y_pred)
    result["Accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
    result["F1"] = f1_score(y_true=y_true, y_pred=y_pred)
    result["FBeta"] = fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.33)
    result["Matthews"] = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    model_result[model_name] = result

    return model_result


def create_area_combinations(areas: list):
    """
    input a list of areas in the city, return all the hold out one possible combinations for source and target areas
    """
    # create area combinations
    seattle_areas = areas
    all_area_combinations = []  # save all the area dictionary in a list
    for test_area in seattle_areas:
        area_combination = {"Source": [], "Target": []}
        for area in seattle_areas:
            if area != test_area:
                area_combination["Source"].append(area)
        area_combination["Target"].append(test_area)
        all_area_combinations.append(area_combination)

    return all_area_combinations


def train_all_data(
    X: pd.DataFrame, X_catboost: pd.DataFrame, cat_feat: list, y: pd.DataFrame
):
    """
    This function putting everything together, create train, test, val, normalize data, evaluate the result
    Input:
        X: features
        X_catboost: dataframe as before but did not encode the categorical feature first for catboost
        cat_feat: list of categorical feature in catboost
        y: target label
    Output:
        a panda dataframe stores the metric result of different algorithm and different area combinations
    """
    ## create the train, val, test
    X_train, y_train, X_valid, y_valid, X_test, y_test = create_train_val_test(X, y)

    ## normalize the numerical features
    X_train_scaled, X_valid_scaled, X_test_scaled = normalize_data(
        X_train=X_train, X_valid=X_valid, X_test=X_test
    )
    ## For catboost create train_val, test
    (
        X_train_cat,
        y_train_cat,
        X_valid_cat,
        y_valid_cat,
        X_test_cat,
        y_test_cat,
    ) = create_train_val_test(X_catboost, y)

    ## train model and save evaluated result
    lr_model, lr_preds = bm.logistic_regression(X_train_scaled, y_train, X_test_scaled)
    lr_result = evaluate_baseline(lr_preds, y_test, "Logistic Regression")

    rf_model, rf_preds = bm.random_forest(X_train_scaled, y_train, X_test_scaled)
    rf_result = evaluate_baseline(rf_preds, y_test, "Random Forest")
    # note catboost does not need scaling, therefore use the original without scalinf, and also input is original cat
    # features https://datascience.stackexchange.com/questions/77312/does-the-performance-of-gbm-methods-profit-from
    # -feature-scaling
    cb_model, cb_preds = bm.catboost(X_train_cat, y_train_cat, cat_feat, X_test_cat)
    cb_result = evaluate_baseline(cb_preds, y_test_cat, "Catboost")

    # result
    output_all_data = {**lr_result, **rf_result, **cb_result}
    df_result_all_areas = pd.DataFrame.from_dict(output_all_data, orient="index")
    return df_result_all_areas


def train_different_areas(
    X_different_areas: pd.DataFrame,
    X_different_areas_catboost: pd.DataFrame,
    cat_feat: list,
    y_different_areas: pd.DataFrame,
):
    """
    This function putting it together, take a list of all the combination of the source and target, as well as the data
    input, standardise the data, train model and evaluate result.

    For the catboost, as it does not need the encoding or normalization, therefore, we just take a dataset without the
    encoding or normalization

    The output is the result of metrics for the combinations of different algorithms on different area_combinations
    """
    # create a dict of area combinations
    seattle_areas = X_different_areas.study_area.unique().tolist()
    all_area_combinations = create_area_combinations(seattle_areas)

    # for each area combination, train different 3 models and save the result
    result_areas = {}
    for area_split in all_area_combinations:
        # create the source data and target data
        X_source, X_target = create_source_target_data(area_split, X_different_areas)
        # refactor below two lines, either merge split or put into the create source target data function
        y_source = y_different_areas.loc[X_source.index]
        y_target = y_different_areas.loc[X_target.index]

        # normalize the data:
        source_scaler = StandardScaler()
        target_scaler = StandardScaler()
        X_source_areas = source_scaler.fit_transform(X_source)
        X_target_areas = target_scaler.fit_transform(X_target)

        # create data for catboost, without the encoding of categorical feature
        X_source_cat, X_target_cat = create_source_target_data(
            area_split, X_different_areas_catboost
        )
        y_source_cat = y_different_areas.loc[X_source_cat.index]
        y_target_cat = y_different_areas.loc[X_target_cat.index]

        # train & transfer the model logistic regression
        lr_model_areas, lr_preds_areas = bm.logistic_regression(
            X_source_areas, y_source, X_target_areas
        )
        lr_result_areas = evaluate_baseline(
            lr_preds_areas, y_target, "Logistic Regression"
        )
        # random forest
        rf_model_areas, rf_preds_areas = bm.random_forest(
            X_source_areas, y_source, X_target_areas
        )
        rf_result_areas = evaluate_baseline(rf_preds_areas, y_target, "Random Forest")

        # catboost classifier, no scaling is needed
        cb_model_areas, cb_preds_areas = bm.catboost(
            X=X_source_cat, y=y_source_cat, cat_feat=cat_feat, X_test=X_target_cat
        )
        cb_result_areas = evaluate_baseline(cb_preds_areas, y_target_cat, "Catboost")

        # merge to one dict
        output = {**lr_result_areas, **rf_result_areas, **cb_result_areas}
        result_areas[str(area_split)] = output

    df_result_10_areas = pd.DataFrame.from_dict(
        {
            (i, j): result_areas[i][j]
            for i in result_areas.keys()
            for j in result_areas[i].keys()
        },
        orient="index",
    )

    return df_result_10_areas


def train_best_model(
    X_different_areas_catboost_100: pd.DataFrame,
    cat_feat: list,
    y_different_areas_100: pd.DataFrame,
):
    """
    This function train the best performing model catboost, and get the feature importance
    """
    # create a dict of area combinations
    seattle_areas = X_different_areas_catboost_100.study_area.unique().tolist()
    all_area_combinations = create_area_combinations(seattle_areas)

    # for each area combination, train different 3 models and save the result
    result_areas = {}
    feat_importance = {}
    for area_split in all_area_combinations:

        # create data for catboost, without the encoding of categorical feature
        X_source_cat, X_target_cat = create_source_target_data(
            area_split, X_different_areas_catboost_100
        )
        y_source_cat = y_different_areas_100.loc[X_source_cat.index]
        y_target_cat = y_different_areas_100.loc[X_target_cat.index]

        # catboost classifier, no scaling is needed
        cb_model_areas, cb_preds_areas = bm.catboost(
            X=X_source_cat, y=y_source_cat, cat_feat=cat_feat, X_test=X_target_cat
        )

        feature_importance = cb_model_areas.get_feature_importance(prettified=True)
        feat_importance[str(area_split)] = feature_importance

        cb_result_areas = evaluate_baseline(cb_preds_areas, y_target_cat, "Catboost")

        # merge to one dict
        output = cb_result_areas
        result_areas[str(area_split)] = output

    df_result_9_areas = pd.DataFrame.from_dict(
        {
            (i, j): result_areas[i][j]
            for i in result_areas.keys()
            for j in result_areas[i].keys()
        },
        orient="index",
    )

    return df_result_9_areas, feat_importance
