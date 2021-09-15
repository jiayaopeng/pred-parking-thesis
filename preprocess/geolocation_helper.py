from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, Polygon
import itertools
from operator import itemgetter
from functools import partial
import pyproj
from shapely.ops import transform
import numpy as np
import pandas as pd
import math

import warnings
from pyprobar import bar, probar

def merge_df_on_nearest_geometries(gdfA, gdfB, gdfB_cols=['street_id']):
    '''
    adds column in geodataframe A with specified columns of clostest entry in geodataframe B
    '''
    gdfA = gdfA.reset_index(drop=True)
    gdfB = gdfB.reset_index(drop=True)
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_crs(epsg=3310).centroid.to_crs(epsg=4326).tolist()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    #compute distance matrix
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf


def define_circle_around_point(lat, lon, m):
    # Azimuthal equidistant projection
    '''
    This function returns for a point specified by lat and long a poylgon describing a circle around that point with radius m meters
    '''
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(m)  # distance in metres
    return transform(project, buf).exterior.coords[:]


def define_circles_around_points(geodataframe, radius):
    """
    For a GeoDataFrame with "geometry" column, define circles with radius m around that points.
    Will print a progressbar (Note: that will not work properly after a kernel interrupt).
    Runtime: 20mins
    """
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    iterator_list = probar(geodataframe.geometry, time_zone="Europe/Berlin")
    geodataframe["geometry"] = [Polygon(define_circle_around_point(lat=coord.y, lon=coord.x, m=radius)) for coord in iterator_list]
    
    return geodataframe


def calculate_diagonal_of_polygon(polygon):
    """
    Calculates the length of the diagonal of a polygon in meters.
    Input geometry should be in crs epsg=4326 (radians):
        Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
        lon1, lat1 = coord1
        lon2, lat2 = coord2
    Source: https://community.esri.com/t5/coordinate-reference-systems/distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """

    # get the maximal points of the polygon
    lon1, lat1, lon2, lat2 = polygon.envelope.bounds  

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters

    return meters
