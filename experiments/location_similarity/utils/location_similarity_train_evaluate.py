import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree as KDTree
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    fbeta_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split

import utils.location_similarity_helper as lsh


def create_cluster_dict(
    cluster_col: list, feature_col: list, target_col: list, data: pd.DataFrame
) -> dict:
    """
    create a dictionary of clusters where key is the cluster label, value is the dataframe of that cluster
    """
    data = data.loc[:, feature_col + cluster_col + target_col]
    dict_of_clusters = {}

    for cluster in np.unique(
        data[cluster_col]
    ):  # for each clustering label of that clustering algorithm
        # if in that cluster dataframe, the availability is either only 0 or 1, then remove that cluster from the dataset
        if (
            len(
                data[(data[cluster_col] == cluster).values][target_col][
                    "availability"
                ].unique()
            )
            == 1
        ):
            continue
        dict_of_clusters[cluster] = data[(data[cluster_col] == cluster).values]

    return dict_of_clusters


def train_model_on_cluster(
    cluster_dict: dict,
    feature_col: list,
    target_col: list,
    cat_features: list,
    iterations=1000,
) -> dict:
    """
    For each cluster in the cluster dictionary, train a model and save the model in a dictionary where the cluster is the key
    """
    matthew_on_train = {}
    models = {}
    feature_importance = {}
    for cluster, cluster_df in cluster_dict.items():
        # change the encoded highway to the str type
        cluster_df["highway"] = cluster_df["highway"].apply(str)
        model = CatBoostClassifier(iterations=iterations)
        model.fit(
            cluster_df[feature_col],
            cluster_df[target_col],
            cat_features=cat_features,
            metric_period=200,
        )
        feat_importance = model.get_feature_importance(prettified=True)
        feature_importance[cluster] = feat_importance

        models[cluster] = model

        pred_on_train = model.predict(cluster_df[feature_col])
        y_true_on_train = cluster_df[target_col]
        matthew_on_train[cluster] = matthews_corrcoef(
            y_true=y_true_on_train, y_pred=pred_on_train
        )

    return models, matthew_on_train, feature_importance


def KLdivergence(x, y) -> int:
    # reference: https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    # from scipy.spatial import cKDTree as KDTree
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, d = x.shape
    m, dy = y.shape
    assert d == dy
    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)
    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    divergence = -np.log(r / s).sum() * d / n + np.log(m / (n - 1.0))

    return divergence


def calculate_distance(
    dict_train: dict,
    dict_test: dict,
    feature_col: list,
    cluster_col: list,
    target_col,
    time_dependent_col: list,
) -> dict:
    """
    This function calculates the distance between all the combinations of train cluster and test cluster
    For each test cluster, we get the distance of between it and train cluster
    Output:
        a dictionary: a mapping of the test cluster label, and its nearest train cluster label based on distance
    """
    result = {}
    # loop through all the test clusters
    features_excluded_for_kl = time_dependent_col + cluster_col + target_col
    for test_cluster, test_value in dict_test.items():
        distances = []
        # loop through all the train clusters
        for train_cluster, train_value in dict_train.items():
            # call the KL divergence function - only select static map data to represent the cluster and to calculate
            # the distance
            distance = KLdivergence(
                # we calculate the KL divergence by street, therefore we need to drop the duplicates so that we have
                # unique street id
                test_value[
                    [col for col in feature_col if col not in features_excluded_for_kl]
                ].drop_duplicates(),
                train_value[
                    [col for col in feature_col if col not in features_excluded_for_kl]
                ].drop_duplicates(),
            )
            # append this pair of distance to list
            distances.append(distance)
        # for test cluster and train cluster distance list, find the min of all the distances
        min_distance_idx = np.argmin(distances)
        # locate the cluster number where the min distance exist
        min_train_cluster_no = list(dict_train.keys())[min_distance_idx]
        # test cluster no vs. smallest distanced train cluster no
        result[test_cluster] = min_train_cluster_no

    return result


def evaluate_result(
    mapping: dict,
    model_dict: dict,
    test_data: dict,
    feature_col: list,
    target_col: list,
) -> dict:
    """
    Train a model on the test cluster with the model built from the its closest(KL divergence) train cluster
    and evaluate the result on 1) the whole test area 2) each of the test cluster in the test area
    Input:
        mapping: a mapping of the test cluster label, and its nearest train cluster label based on distance
        model: a dictionary of each train cluster and the model object
        test: a dictionary of test clusters where key is cluster label and value is the dataframe of that cluster
              for the case where we check overfit, the key is the train cluster label instead of test cluster label
        feature_col: feature columns
        target col: only one target but put it into a list
    Output:
        a dictionary of different evaluation metrics for that cluster
        a dictionary of matthews for different test cluster in the test area where key is the label of test cluster,
        value is the matthew score
    """
    preds = np.asarray([])
    y_trues = np.asarray([])
    # below two dictionary used to save result per test cluster
    preds_dict = {}
    y_trues_dict = {}
    # matthew_different_test_cluster = {}
    for cluster_test, cluster_df in test_data.items():
        # make prediction and evaluate
        # change the cluster highway to string
        cluster_df["highway"] = cluster_df["highway"].apply(str)
        # use cluster_test key to get its nearest model's cluster label, then use it locate the model in the model disctionary
        # for pred and y true, for each test cluster, concatenate their y_true and preds along the rows
        pred = model_dict[mapping[cluster_test]
                          ].predict(cluster_df[feature_col])
        preds_dict[cluster_test] = pred  # save result per cluster
        preds = np.concatenate([preds, pred], axis=0)

        # get the y_true
        y_trues_dict[cluster_test] = cluster_df[target_col].values.reshape(-1)
        y_trues = np.concatenate(
            [y_trues, cluster_df[target_col].values.reshape(-1)], axis=0
        )

    # save the result for per cluster in the test area, where the key is the test cluster no and value is the score
    matthews_dict = {}
    for (test_cluster_no1, pred), (test_cluster_no2, y_true) in zip(
        preds_dict.items(), y_trues_dict.items()
    ):
        if test_cluster_no1 == test_cluster_no2:  # do a test
            matthews_dict[test_cluster_no1] = matthews_corrcoef(
                y_true=y_true, y_pred=pred
            )

    # evaluate the result for model transfer on the whole test area(concat the result from different test clusters)
    auc = roc_auc_score(y_score=preds, y_true=y_trues)
    recall = recall_score(y_true=y_trues, y_pred=preds)
    precision = precision_score(y_true=y_trues, y_pred=preds)
    accuracy = accuracy_score(y_true=y_trues, y_pred=preds)
    f1 = f1_score(y_true=y_trues, y_pred=preds)
    fbeta = fbeta_score(y_true=y_trues, y_pred=preds, beta=0.33)
    matthews = matthews_corrcoef(y_true=y_trues, y_pred=preds)

    # zip the result and var name and convert it to dict
    score_ls = [auc, recall, precision, accuracy, f1, fbeta, matthews]
    score_names = ["AUC", "Recall", "Precision",
                   "Accuracy", "F1", "FBeta", "Matthews"]
    zip_score_ls_names = zip(score_names, score_ls)
    score_dict = dict(zip_score_ls_names)

    return score_dict, matthews_dict


def evaluate_valid_result(
    model_dict: dict, valid_data: dict, feature_col: list, target_col: list
) -> dict:
    """
    This main purpose of this function is to evaluate the overfitting situation within training cluster by evaluate the
    models which are trained training data, and predict on the valid data(NOTE: take the training data as whole and split
    the training data into train and test
    """

    # below two dictionary used to save result per test cluster
    preds_dict = {}
    y_trues_dict = {}
    # matthew_different_test_cluster = {}
    for cluster_train_no, cluster_df in valid_data.items():
        # make prediction and evaluate
        # change the cluster highway to string
        cluster_df["highway"] = cluster_df["highway"].apply(str)
        # use model from the same key in the model dict(where we stored by the cluster no) to predict the data
        preds_dict[cluster_train_no] = model_dict[cluster_train_no].predict(
            cluster_df[feature_col]
        )
        # get the y_trues
        y_trues_dict[cluster_train_no] = cluster_df[target_col].values.reshape(
            -1)

        # combine the above as one dictionary
        valid_matthews = {}
        for (cluster_train_no1, pred), (cluster_train_no2, y_true) in zip(
            preds_dict.items(), y_trues_dict.items()
        ):
            if cluster_train_no1 == cluster_train_no2:
                valid_matthews[cluster_train_no1] = matthews_corrcoef(
                    y_true=y_true, y_pred=pred
                )

    return valid_matthews


def create_cluster_dictionary(
    cluster_col: list,
    feature_col: list,
    target_col: list,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
):
    """
    This function takes the training and test data, preprocess the data, in the end generate a dictionary of of clusters
    for train and test,
        1) encode the categorical variable highway
        2) normalize the data
        3) merge it back with the data train to get the target column and also the cluster label
        3) call the create dictionary of cluster function
    Output:
        train and test dictionary, where key is the cluster label, and value is the dataframe
    """
    data_train = data_train.set_index(
        ["street_id", "observation_interval_start"])
    data_test = data_test.set_index(
        ["street_id", "observation_interval_start"])

    cat_features = ["hour", "weekday", "highway"]

    # encode highway - only for KL divergence between clusters calculation because we need it normalized
    encoded_data_train, encoder = lsh.encode_categorical(
        encoder="target_encoder",
        col_encoded=["highway"],
        feature=data_train[feature_col],
        target=data_train[target_col],
    )
    # use the encoder to transform the test
    encoded_data_test = encoder.transform(
        X=data_test[feature_col], y=data_test[target_col]
    )

    # normalize the train and test data's numerical features, take the whole features
    scaler = StandardScaler()
    scaled_data_train = encoded_data_train.copy(
        deep=True
    )  # copy the data so we do not change the original
    scaled_data_train[
        [x for x in feature_col if x not in cat_features]
    ] = scaler.fit_transform(
        scaled_data_train[[x for x in feature_col if x not in cat_features]]
    )

    # columns=encoded_data_train.columns,
    # index=encoded_data_train.index
    scaled_data_test = encoded_data_test.copy(deep=True)
    scaled_data_test[
        [x for x in feature_col if x not in cat_features]
    ] = scaler.transform(
        scaled_data_test[[x for x in feature_col if x not in cat_features]]
    )

    # merge the cluster col and target column back to the transformed data
    scaled_merged_data_train = pd.merge(
        scaled_data_train,
        data_train[cluster_col + target_col],
        left_index=True,
        right_index=True,
    )
    scaled_merged_data_test = pd.merge(
        scaled_data_test,
        data_test[cluster_col + target_col],
        left_index=True,
        right_index=True,
    )

    # create cluster based on the scaled data, later used for calculate KL divergence
    train_dict = create_cluster_dict(
        cluster_col, feature_col, target_col, scaled_merged_data_train
    )
    test_dict = create_cluster_dict(
        cluster_col, feature_col, target_col, scaled_merged_data_test
    )

    return train_dict, test_dict


def cluster_train_evaluate_result(
    cluster_col: list,
    feature_col: list,
    target_col: list,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    iterations=1000,
) -> dict:
    """
    This function wraps up several main steps together:
        1) call create dictionary of clusters
        2) call calculate distance map between based on KL divergence
        3) for each test cluster, find its "closest" train cluster
        4) call the train model on cluster function
        5) call the evaluate result function
    Beyond that, the function also returns values to enable the check on over fitting issues by:
        1) return the matthew and train cluster size correlation
        2) calculate the matthew on train and matthew on test (if there is overfitting in terms of transfer)
        3) random split the original train data into 80-20 percent(train and val), and check the matthews(if there is
        overfitting within train cluster)
        ----> from above, we wish to conclude if our method fails due to domain shift or due to overfitting
    """
    time_dependent_features = [
        "current_capacity",
        "tempC",
        "windspeedKmph",
        "precipMM",
        "hour",
        "weekday",
        "ongoing_trans",
    ]
    # time_dependent_features = ['current_capacity', 'tempC', 'windspeedKmph', 'precipMM', 'hour', 'weekday']
    cat_features = ["hour", "weekday", "highway"]

    # create cluster dictionary
    train_clusters_dict, test_clusters_dict = create_cluster_dictionary(
        cluster_col, feature_col, target_col, data_train, data_test
    )
    # get the train-test cluster map(each test cluster, find its closest train cluster)
    # key: is the test cluster
    # value: is the cluster no. of the current test cluster's closest cluster in train/source data
    distance_map = calculate_distance(
        train_clusters_dict,
        test_clusters_dict,
        feature_col,
        cluster_col,
        target_col,
        time_dependent_features,
    )

    closest_clusters = {}  # locate the dataframe of the closest train cluster
    closest_train_cluster_data_sizes = (
        {}
    )  # test cluster number as key, train cluster data size as value
    # loop through the training clusters
    for train_cluster_no, cluster_data in train_clusters_dict.items():
        # if we could find the number training cluster in our distance map
        for test_cluster_no, tmp_train_cluster_no in distance_map.items():
            if tmp_train_cluster_no != train_cluster_no:
                continue
            closest_clusters[train_cluster_no] = cluster_data
            closest_train_cluster_data_sizes[train_cluster_no] = len(
                cluster_data)
            print(f"There are {len(cluster_data)} datapoint in the closest")

    ##############################
    ## Overfitting within Train ##
    ##############################
    # split the matched closest train cluster into train and valid(to valid if there is overfitting
    closest_clusters_train_80 = {}
    closest_clusters_val_20 = {}
    for train_cluster_no, train_cluster_df in closest_clusters.items():
        train_cluster_df_80, valid_cluster_df_20 = train_test_split(
            train_cluster_df, test_size=0.2, random_state=42
        )
        closest_clusters_train_80[train_cluster_no] = train_cluster_df_80
        closest_clusters_val_20[train_cluster_no] = valid_cluster_df_20
    # train the model in the splited training data, and get the model dictionary: return model trained, return matthew trained and predicted on train data
    model_dict_train_80, matthew_train_80, _ = train_model_on_cluster(
        closest_clusters_train_80,
        feature_col,
        target_col,
        cat_features,
        iterations=iterations,
    )

    ####################
    ## Model Transfer ##
    ####################
    # In all the training clusters, only train a model on the closest cluster to the current test cluster
    # (Namely, do not train one model on every train clusters)
    (
        model_dict,
        matthews_per_train_cluster,
        feat_importance_per_train_cluster,
    ) = train_model_on_cluster(
        closest_clusters, feature_col, target_col, cat_features, iterations=iterations
    )

    # evaluate the result of model transfer
    scores, matthews_per_test_cluster = evaluate_result(
        distance_map, model_dict, test_clusters_dict, feature_col, target_col
    )

    # put the feature importance per test cluster together, by using the distance map to match test cluster labels
    feat_importance_dict = {}
    for test_cluster_no, train_cluster_no in distance_map.items():
        feat_importance_dict[test_cluster_no] = feat_importance_per_train_cluster[
            distance_map[test_cluster_no]
        ]

    # Evaluate overfitting (matthew and training size correlation)
    matthew_train_size_correlation = {}
    for test_cluster_no, matthew in matthews_per_test_cluster.items():
        matthew_train_size_correlation[test_cluster_no] = {
            "matthew": matthew,
            # use the test cluster no. to locate the train cluster no and then to locate
            "train_cluster_size": closest_train_cluster_data_sizes[
                distance_map[test_cluster_no]
            ],
        }

    # Evaluate overfiting (training cluster and test cluster overfiting)
    matthew_train_test = {}
    for test_cluster_no, matthew in matthews_per_test_cluster.items():
        matthew_train_test[test_cluster_no] = {
            "train_cluster_matthew": matthews_per_train_cluster[
                distance_map[test_cluster_no]
            ],
            "test_cluster_matthew": matthews_per_test_cluster[test_cluster_no],
        }

    # Evaluate overfiting(within training clusters overfitting)
    matthew_valid_20 = evaluate_valid_result(
        model_dict_train_80, closest_clusters_val_20, feature_col, target_col
    )
    # put the two dictionary together
    matthew_train_val = {}
    for (cluster_train_no1, matthew_train), (cluster_train_no2, matthew_val) in zip(
        matthew_train_80.items(), matthew_valid_20.items()
    ):
        if cluster_train_no1 == cluster_train_no2:
            matthew_train_val[cluster_train_no1] = {
                "train_cluster_matthew_80": matthew_train_80[cluster_train_no1],
                "valid_cluster_matthew_20": matthew_valid_20[cluster_train_no1],
            }
    # put the two overfit together(1) check the overfit within train 2) the overfit between train and test)
    overfit_final_dict = {}
    for test_cluster_no, train_cluster_no in distance_map.items():
        train_val_info = matthew_train_val[train_cluster_no]
        train_test_info = matthew_train_test[test_cluster_no]
        overfit_final_dict[test_cluster_no] = {
            **train_val_info, **train_test_info}
    print(overfit_final_dict)

    return (
        scores,
        matthew_train_size_correlation,
        overfit_final_dict,
        feat_importance_dict,
    )


def train_evaluate_all_approaches(
    cluster_cols: list,
    feature_col: list,
    target_col: list,
    df_train_clusters: pd.DataFrame,
    df_test_clusters: pd.DataFrame,
    iterations=1000,
) -> dict:
    """
    For different cluster labels generated by different clustering algorithm, calculate the result of different
    evaluation metrics
    Output:
        a nested dictionary where {cluster_label {'metric':, value of that metric}}
    """
    result_feat_importance = {}
    result_matthew_size_corr = (
        {}
    )  # save to see the correlation between training data size and matthew
    result_matthew_overfit = {}  # save overfit
    result = {}
    for cluster_col in cluster_cols:
        train_input = df_train_clusters
        test_input = df_test_clusters

        if (cluster_col == ["db_cluster_label_gps"]) or (
            cluster_col == ["db_cluster_label_sim"]
        ):
            # if the cluster label is of DB scan, then for the model, remove where cluster label = -1 from train and
            # test data set
            train_input = df_train_clusters[
                (df_train_clusters.db_cluster_label_gps != -1)
            ]
            test_input = df_test_clusters[(
                df_test_clusters.db_cluster_label_gps != -1)]

        (
            scores,
            matthew_train_size_corr,
            matthew_overfit,
            feature_importance,
        ) = cluster_train_evaluate_result(
            cluster_col,
            feature_col,
            target_col,
            train_input,
            test_input,
            iterations=iterations,
        )

        # [0] is due to the fact that['cluster_label'] is a list
        result[str(cluster_col[0])] = scores
        result_matthew_size_corr[str(cluster_col[0])] = matthew_train_size_corr
        result_matthew_overfit[str(cluster_col[0])] = matthew_overfit
        result_feat_importance[str(cluster_col[0])] = feature_importance

    return (
        result,
        result_matthew_size_corr,
        result_matthew_overfit,
        result_feat_importance,
    )
