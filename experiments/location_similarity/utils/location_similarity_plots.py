import folium
import geopy
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import random
import logging
import seaborn as sns


def plot_cluster_folium(
    data: pd.DataFrame,
    train=True,
    study_area=None,
    cluster_label=None,
    tiles="OpenStreetMap",
    city="Seattle",
):
    """
    Plot the clusters on the folium map:
        Input a list of target city's longetude and lattitude, a dataframe with longitude and latitude column, colored the map with clusters
        train: True by default, change the name of the output map
    """
    # get the city lon and lat as a list
    city_ = city
    locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
    city_lon_lat = locator.geocode(city_)
    city_lon_lat = [city_lon_lat.latitude, city_lon_lat.longitude]

    m = folium.Map(city_lon_lat, tiles=tiles, zoom_start=12)
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "pink",
        "gray",
        "black",
        "yellow",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "lightblue",
        "lightgreen",
        "lightgray",
    ]
    # if cluster -1 in the case of DBscan means the points are outliers, only plot max 10 clusters
    if not cluster_label:
        data.apply(
            lambda row: folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=1,
                weight=2,
                fill=True,
                fill_color=colors[int(data.loc[row.name, study_area]) % len(colors)],
                color=colors[int(data.loc[row.name, study_area]) % len(colors)],
            ).add_to(m),
            axis=1,
        )

        m.save("map.html")
    else:
        # only plot max 10 clusters and do not plot the outlier
        data[(data[cluster_label] != -1) & (data[cluster_label] < 10)].apply(
            lambda row: folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=1,
                weight=2,
                fill=True,
                fill_color=colors[data.loc[row.name, cluster_label] % len(colors)],
                color=colors[data.loc[row.name, cluster_label] % len(colors)],
            ).add_to(m),
            axis=1,
        )
        # mark the map with train and test
        if train == True:
            plot_name = f"cluster map for training cluster {cluster_label}"
        else:
            plot_name = f"cluster map for test cluster {cluster_label}"

        m.save(plot_name + ".html")


def plot_dendrogram(model, **kwargs):
    # ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(15, 10))
    plt.xlabel("Number of points in node")
    plt.ylabel("Measure of dissimilarity of the node")
    dendrogram(linkage_matrix, **kwargs)
    plt.show()


def plot_highway(data: pd.DataFrame):
    """
    This function is to analysis the distribution of the highway variable
    Input:
        data: the raw features
    output:
        a barplot
    """
    df_street_highway = data.loc[:, ["highway", "capacity"]]
    # subset the data, keep only one record per street
    df_street_highway_capa = df_street_highway[
        ~df_street_highway.index.duplicated(keep="first")
    ]
    df_street_highway_capa["street_id"] = df_street_highway_capa.index

    df_street_highway_capa_groupped = (
        df_street_highway_capa.groupby(["highway"])
        .agg({"street_id": "count", "capacity": "sum"})
        .reset_index()
        .rename(
            columns={"street_id": "no. of streets", "capacity": "all_capacity_total"}
        )
    )

    df_street_highway_capa_groupped["mean_capacity"] = (
        df_street_highway_capa_groupped["all_capacity_total"]
        / df_street_highway_capa_groupped["no. of streets"]
    )

    plt.figure(figsize=(15, 10))
    plt.bar(
        df_street_highway_capa_groupped["highway"],
        df_street_highway_capa_groupped["no. of streets"],
    )
    plt.show()


def plot_distance_matrix(
    distance_matrix1: pd.DataFrame,
    distance_matrix2: pd.DataFrame,
    no_of_street: int,
    distance_matrix1_title: str,
    distance_matrix2_title: str,
):
    """
    This function randomly select n number of  streets'similarity and plot the similarity on the two heat map side by side
    """

    # select index
    ls_idx = list(distance_matrix1.index)
    sampled_idx = random.sample(ls_idx, no_of_street)
    logging.info("The sampled indexes of the streets are {sampled_idx}")
    # loc distance_matrix1 data based on index
    temp = distance_matrix1[distance_matrix1.index.isin(sampled_idx)]
    sampled_pair_dist = temp[temp.columns.intersection(sampled_idx)]
    # log distance_matrix2 data based on index
    temp2 = distance_matrix2[distance_matrix2.index.isin(sampled_idx)]
    sampled_pair_dist2 = temp2[temp2.columns.intersection(sampled_idx)]

    # plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    ax1.title.set_text(distance_matrix1_title)
    sns.heatmap(sampled_pair_dist, cmap="OrRd", ax=ax1)
    ax2.title.set_text(distance_matrix2_title)
    sns.heatmap(sampled_pair_dist2, cmap="OrRd", ax=ax2)
    plt.show()


def plot_correlation_distance(correlation_matrix, title: str):
    """
    Input: correlation matrix
           title of the plot
    """
    plt.hist(correlation_matrix)
    plt.title(title)
    plt.xlim(-1, 1)
    plt.grid()
    plt.show()
