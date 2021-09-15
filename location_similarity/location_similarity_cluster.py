import numpy as np
import pandas as pd
import time
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering


def canopy(X, T1, T2, distance_metric='euclidean', filemap=None):
    # ref: https://gist.github.com/gdbassett/528d816d035f2deaaca1
    # TODO: try cannopy for GPS and streetsim
    """
    For our settings, we set T1=T2, also need to divide by the radian
    Example:
            T1 = 0.5/km_per_radian
            T2 = T1
    """
    # train cannopy
    canopies = dict()
    X1_dist = pairwise_distances(X, metric=distance_metric)
    canopy_points = set(range(X.shape[0]))
    while canopy_points:
        point = canopy_points.pop()
        i = len(canopies)
        canopies[i] = {"c": point, "points": list(
            np.where(X1_dist[point] < T2)[0])}
        canopy_points = canopy_points.difference(
            set(np.where(X1_dist[point] < T1)[0]))
    if filemap:
        for canopy_id in canopies.keys():
            canopy = canopies.pop(canopy_id)
            canopy2 = {"c": filemap[canopy['c']], "points": list()}
            for point in canopy['points']:
                canopy2["points"].append(filemap[point])
            canopies[canopy_id] = canopy2

    # get all the clusters and labels from canopy
    cp_gps_clusters = [[key for key, val in canopies.items() if i in val['points']]
                       for i in range(len(X))]
    # for cannopy some points belong to two clusters, only take the first one
    cannopy_labels_gps = [i[0] for i in cp_gps_clusters]

    return cannopy_labels_gps, canopies


def db_scan(data: pd.DataFrame, eps: int, min_samples: int, algorithm='ball_tree', metric='haversine'):
    # TODO: rename the second para as len(data) for all similar function
    """
    When use DB scan for GPS clusters, we need to divide the eps by km_per_radian = 6371.0088
        example:
            km_per_radian = 6371.0088
            eps = 0.2 / km_per_radian
    When use DB scan for similarity clusters, we could solution increase eps, decrease min_samples
    Output: db_labels: labels of the clustering
            db_num_clusters: how many clusters are there
            db: the clustering model

    """
    # use DB scan to cluster the points, code ref: https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    start_time = time.time()
    # why below convert the angles to radian works (angle of latitude, could get the radian)
    db = DBSCAN(eps=eps, min_samples=min_samples,
                algorithm=algorithm, metric=metric).fit(data)
    # get the cluster labels for each coordinates
    db_labels = db.labels_

    # get the number of clusters
    db_num_clusters = len(set(db_labels))

    # all done, print the outcome
    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
    print(message.format(len(data), db_num_clusters, 100 * (1 - float(db_num_clusters) / len(data)),
                         time.time() - start_time))
    print('Silhouette coefficient: {:0.03f}'.format(
        metrics.silhouette_score(data, db_labels)))

    return db_labels, db_num_clusters, db


def kmeans(data, n_clusters, random_state=42):
    start_time = time.time()

    # use simple kmeans with L2 distance
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    kmeans_labels = kmeans.labels_

    # get the number of clusters
    kmeans_num_clusters = len(set(kmeans_labels))

    # all done, print the outcome
    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
    print(
        message.format(len(data), kmeans_num_clusters, 100 * (1 - float(kmeans_num_clusters) / len(data)),
                       time.time() - start_time))
    print('Silhouette coefficient: {:0.03f}'.format(
        metrics.silhouette_score(data, kmeans_labels)))

    return kmeans_labels, kmeans_num_clusters, kmeans


def agg_clustering(
    data,
    n_clusters,
    linkage='ward',
    compute_distances=True
):
    start_time = time.time()

    agg = AgglomerativeClustering(
        n_clusters=n_clusters, compute_distances=compute_distances).fit(data)
    agg_labels = agg.labels_

    agg_num_clusters = agg.n_clusters_

    message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
    print(message.format(len(data), agg_num_clusters, 100 * (1 - float(agg_num_clusters) / len(data)),
                         time.time() - start_time))
    print('Silhouette coefficient: {:0.03f}'.format(
        metrics.silhouette_score(data, agg_labels)))

    return agg_labels, agg_num_clusters, agg