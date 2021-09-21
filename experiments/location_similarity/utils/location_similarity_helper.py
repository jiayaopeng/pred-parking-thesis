import time
import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import category_encoders as ce
import math
import json

import experiments.location_similarity.utils.location_similarity_cluster as lsc


def fix_maxspeed(data: pd.DataFrame) -> pd.DataFrame:
    """
    The old seattle_train_here_data's maxspeed column is problematic, this func will only be used until the bug fixed
    """
    if len(data[data["maxspeed"].str.contains("\['", na=False)]) != 0:
        data.loc[:, "maxspeed"] = data.loc[:, "maxspeed"].str.strip("['")
        # assert the stripped result
        assert len(data[data["maxspeed"].str.contains("\['", na=False)]) == 0
        logging.info(
            "old data with bugged maxspeed data is still used, string has been stripped"
        )
        return data
    else:
        return data


def preprocess_for_similarity_analysis(
    data: pd.DataFrame,
    selected_feature_names: list,
    options={
        "impute_maxspeed": False,
        "encode_highway": False,
        "time_dependant_features": None,
    },
) -> pd.DataFrame:
    """
    This function does following things:
    1) select features (if include time dependent features)
    2) encode categorical data
    3) impute maxspeed data
    4) separate geometry data and feature data
    Input:
        data  dataset with all the features
        selected_feature_names:  a list of feature names which are not time-dependent
        options.maxspeed: if maxspeed is True, then we use the column, and vice versa
        options.time_dependent_features:  a list of feature names which are time-dependent
        **otherparams  optional params to be added for time_dependent_features

    Output:
        df_features  dataframe which has been cleaned(encoded, imputed and so on) identified by street_id
        df_geometry  retain the geometry column with street_id

    """

    final_feature_cols = selected_feature_names
    if options["time_dependant_features"]:
        final_feature_cols = selected_feature_names + options["time_dependant_features"]
        # TODO: below part will be filled when in the second iteration we decide to use time-dependent features
        # and for now just a placeholder
        pass
    # select the data based on the final_feature_cols
    features = data.loc[:, final_feature_cols]

    # prepare encoding data input
    geometries = features.loc[:, ["geometry", "street_id"]]
    features = features.set_index("street_id")

    # if we later decide to use max_speed feature,then we need to impute the missing values
    if options["impute_maxspeed"]:
        # call the maxspeed imputation function
        imputed_features = impute_maxspeed(features)
        X = imputed_features.drop(["availability", "geometry"], axis=1)
    else:
        X = features.drop(["availability", "geometry", "maxspeed"], axis=1)
    y = features[["availability"]]

    # call function encode the categorical variable highway
    if options["encode_highway"]:
        similarity_features, _ = encode_categorical("target_encoder", ["highway"], X, y)
    else:
        similarity_features = X.drop(["highway"], axis=1)

    # TODO: check the same street different timestamps's highway column are encoded with same numeric value

    return features, similarity_features, y, geometries


def impute_maxspeed(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function impute maxspeed column with defined method
    """

    # motorway_link, use maximum speed of the maxspeed
    data.loc[
        (data["maxspeed"].isnull()) & (data["highway"] == "motorway_link"), "maxspeed"
    ] = (data["maxspeed"].astype(float).max())

    # residential, use mode of the residential speed
    mode = float(data.loc[data["highway"] == "residential"]["maxspeed"].mode()[0])
    data.loc[
        (data["maxspeed"].isnull()) & (data["highway"] == "residential"), "maxspeed"
    ] = mode

    # living street, use min of the maxspeed column
    data.loc[
        (data["maxspeed"].isnull()) & (data["highway"] == "living_street"), "maxspeed"
    ] = (data["maxspeed"].astype(float).min())

    return data


def encode_categorical(
    encoder: str, col_encoded: list, feature: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    """
    This function encode the categorical column with user defined method, a wrapper to use the categorical encoder package
    """
    if encoder == "target_encoder":
        ce_target_encoder = ce.TargetEncoder(cols=col_encoded)
        similarity_features = ce_target_encoder.fit_transform(feature, target)
        return similarity_features, ce_target_encoder

    elif encoder == "onehot":
        onehot_encoder = ce.onehot.OneHotEncoder(cols=col_encoded)
        similarity_features = onehot_encoder.fit_transform(feature, target)
        return similarity_features, onehot_encoder

    else:
        # TODO: other encoder to be tested out, below only as placeholder
        pass


def street_pairwise_dist(similarity_features, metric: str) -> pd.DataFrame:
    """
    This function takes the similarity features and calculate the pairwise distance
    Input:
        similarity_feature:  input feature set, could be normalized or un normalized depends on the metric
        metric: which distance metric to choose
    Output:
        distance_matrix: a panda dataframe with all the paired distance
    """
    distance_matrix = pd.DataFrame(
        squareform(pdist(similarity_features, metric=metric)),
        columns=similarity_features.index,
        index=similarity_features.index,
    )
    return distance_matrix


def scale_before_plot_correlation(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    This function will be called before plot the correlation matrix to make the values of euclinean distance and cosine distance in the same range[0,1]
    """
    # get the column max value
    column_maxes = distance_matrix.max()
    # max of column maxes
    df_max = column_maxes.max()

    # get min
    column_mins = distance_matrix.min()
    # min of column min
    df_min = column_mins.min()

    # scale
    df_min_max_scaled = (distance_matrix - df_min) / (df_max - df_min)

    return df_min_max_scaled


def distance_between_coordinates(coordinate1, coordinate2):
    """
    Calculates the length of two coordinate points in meters.
    Input geometry should be in crs epsg=4326 (radians):
        Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
        lon1, lat1 = coord1
        lon2, lat2 = coord2
    Source: https://community.esri.com/t5/coordinate-reference-systems/distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """

    # get the maximal points of the polygon
    lon1, lat1, lon2, lat2 = coordinate1.x, coordinate1.y, coordinate2.x, coordinate2.y

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters

    return meters


def calculate_street_similarity_matrix(gdf: pd.DataFrame) -> object:
    # create dictionary of dictionary to hold the data each street has a dictionary
    # of distances compared with all the streets
    # TODO: (nice to have) refactor below part with apply in a for loop
    start_time = time.time()
    result = {}
    for i in range(0, len(gdf)):
        street_to_compare = gdf.iloc[[i]]
        str1 = street_to_compare.index.astype("str")[0]
        # For each street_id, get open an empty dict, the street to compare
        if str1 not in result:
            result[str1] = {}

        for j in range(0, len(gdf)):
            street = gdf.iloc[[j]]
            str2 = street.index.astype("str")[0]
            # open any dict, for street to be compared
            if str2 not in result:
                result[str2] = {}
            # do not calculate the (x,y)(y,x) case, because they are essential with same distance
            if str2 in result[str1] or str1 in result[str2]:
                continue
            # call the distance calculation function
            d = distance_between_coordinates(
                street_to_compare.line_centroid.values, street.line_centroid.values
            )
            result[str1][str2] = d
            result[str2][str1] = d
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)
    return result


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


def get_cluster_size_for_areas(area_input_data):
    area_cluster_size = {}
    min_street_count_in_cluster = 25

    for area_name in list(area_input_data.keys()):
        area_cluster_size[area_name] = {}
        area_street_count = len(area_input_data[area_name]["Target"])
        remaining_area_street_count = len(area_input_data[area_name]["Source"])
        area_cluster_count = max(
            1, round(area_street_count / min_street_count_in_cluster)
        )
        remaining_area_cluster_count = round(
            remaining_area_street_count / min_street_count_in_cluster
        )
        area_cluster_size[area_name]["Source"] = remaining_area_cluster_count
        area_cluster_size[area_name]["Target"] = area_cluster_count
        print(
            area_name,
            ": ",
            area_street_count,
            "/",
            remaining_area_street_count,
            " -- ",
            area_cluster_count,
            "/",
            remaining_area_cluster_count,
        )

    return area_cluster_size


def create_cluster_label(
    area_input_data,
    df_pair_dist_max_normalized,
    area_name: str,
    base: str,
    algorithm: object,
    data: any,
    is_train: bool,
) -> any:
    """
    This function takes the similarity metrics, the algorithm and if using the training data, and return the call of the clustring algo
    Input:
        base: if we are clustering based on GPS or based on vector street similarity
        algorithm: the clustering algorithm includes DB scan, Kmeans, hierarchical agglomerative clustering(AGG)
        is_train: takes boolean to determine if the data is trainning or test, because we initializing sepearate clustering process in train and test
    Output:
        by if else condition, call the clusteting algo
    """
    with open("config/clustering_params.json") as json_file:
        # the config file path is relative to the file using it
        cluster_input = json.load(json_file)

    area_cluster_size = get_cluster_size_for_areas(area_input_data)

    config = cluster_input[base][("train" if is_train else "test")][algorithm]

    n_cluster = area_cluster_size[area_name]["Source" if is_train else "Target"]

    if n_cluster == 1:
        return np.full(len(data), -1)

    if base == "sim":
        study_area_streets = data[["study_area"]]
        input_data = df_pair_dist_max_normalized.filter(
            items=study_area_streets.index
        ).filter(items=study_area_streets.index, axis=0)
    else:
        study_area_street_coords = data[["lon", "lat"]]
        input_data = study_area_street_coords.to_numpy()
        if algorithm == "db_scan":
            input_data = np.radians(input_data)

    if algorithm == "db_scan":
        if base == "sim":
            return lsc.db_scan(
                data=input_data,
                min_samples=int(config["min_samples"]),
                eps=float(config["eps"]),
                algorithm=config["algorithm"],
                metric=config["metric"],
            )[0]
        elif base == "gps":
            return lsc.db_scan(
                data=input_data,
                min_samples=int(config["min_samples"]),
                eps=float(config["eps"])
                / km_per_radian,  # for DB scan, when clustering GPS, we have to divide by the km per radian
            )[0]
    elif algorithm == "kmeans":
        return lsc.kmeans(data=input_data, n_clusters=n_cluster)[0]
    elif algorithm == "agg_clustering":
        return lsc.agg_clustering(data=input_data, n_clusters=n_cluster)[0]
