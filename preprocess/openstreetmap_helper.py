import folium
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from geolocation_helper import define_circles_around_points



def print_pbp_locations_on_map(pbp_location_geometries, paybyphone_data, location=None):
    """
    Visualize points of parking events and corresponding location on OpenStreetMap.
    
    :param: pbp_location_geometries: PayByPhone parking location geometries
    :param paybyphone_data: transaction event data with geometries
    """
    
    if not location:
        # Seattle
        location=[47.65, -122.35]

    m = folium.Map(location=location, zoom_start=12, tiles="OpenStreetMap")

    col_counter = 0
    row_prev = None
    for index, row in paybyphone_data.iterrows():

        tooltip = str(row["action"]) + " at " + str(pd.to_datetime(row["created"], unit="ms")) + " at journey " + str(row["journeyId"])

        # colors = ['red', 'darkpurple', 'orange', 'lightgray', 'darkred', 'lightblue', 'blue', 'darkgreen', 'white', 'black', 'lightgreen', 'beige', 
        #           'green', 'pink', 'gray', 'darkblue', 'cadetblue', 'purple', 'lightred']
        
        folium.Marker([row["lat"], row["long"]], tooltip=tooltip, icon=folium.Icon(color="darkgreen")).add_to(m)

        if row["advLocationId"]:
            try:
                folium.Choropleth(pbp_location_geometries[pbp_location_geometries.advLocationId == str(row["advLocationId"])], line_weight=8, line_color="blue",tooltip=tooltip).add_to(m)
            except:
                pass

    return m


def retrieve_pois(place_name="Seattle, US"):
    """
    Retrieve point of interests from OpenStreetMap module "geometries".
    Runtime: 5mins
    """
    ox.config(timeout=100000)

    # Splitting OSM API call into 3 calls to avoid running into API Timeout
    pois_list = []
    
    # Extract POIs from Open Street Map
    print("Retrieving highway...")
    pois = ox.geometries_from_place(place_name, tags={'highway':True})
    pois_list.append(pois)

    print("Retrieving landuse...")
    pois = ox.geometries_from_place(place_name, tags={'landuse':True})
    pois_list.append(pois)

    print("Retrieving building...")
    pois = ox.geometries_from_place(place_name, tags={'building':True})
    pois_list.append(pois)
    pois = pd.concat(pois_list)

    # Presentation code
    pois.to_crs(epsg=4326, inplace=True)

    fig,ax = plt.subplots(figsize=(16,16))
    pois[pois['landuse'].notna()].plot(ax=ax, column='landuse', cmap='tab20b', legend=True)
    pois[pois['highway'].notna()].plot(ax=ax, color='white')
    pois[pois['building'].notna()].plot(ax=ax, color='dimgrey', zorder=3)
    ax.set_facecolor('darkgrey')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    # return the POIs as a single concattenated DataFrame
    return pois


def cluster_pois(pois):
    
    """
    Filter for interesting POI types and cluster into commercial, residential, transportation, school, eventsite.
    Return the centroids in column "geometry".
    """
    
    pois_filtered = pois[["geometry", "landuse", "building"]].dropna(subset=["landuse", "building"], how="all")

    # Clustering the tags into commercial, residential, transportation, school, eventsite
    commercial_tags = ["industrial", "retail", "office", "commercial", "hotel", "hospital", "government", "warehouse", "central_office"]
    pois_filtered["commercial"] = [1 if (row.landuse in commercial_tags or row.building in commercial_tags) else 0 
                                           for index, row in pois_filtered.iterrows()]

    residential_flags = ["residential", "recreation_ground", "apartments", "garages", "house", "civic", "garage"]
    pois_filtered["residential"] = [1 if (row.landuse in residential_flags or row.building in residential_flags) else 0 
                                           for index, row in pois_filtered.iterrows()]

    transportation_flags = ["railway", "train_station", "transportation", "railway_station"]
    pois_filtered["transportation"] = [1 if (row.landuse in transportation_flags or row.building in transportation_flags) else 0 
                                           for index, row in pois_filtered.iterrows()]

    school_flags = ["university", "school", "kindergarten", "college"]
    pois_filtered["schools"] = [1 if (row.landuse in school_flags or row.building in school_flags) else 0 
                                           for index, row in pois_filtered.iterrows()]

    eventsite_flags = ["plaza", "churchyard", "church", "cathedral", "temple", "stadium"]
    pois_filtered["eventsites"] = [1 if (row.landuse in eventsite_flags or row.building in eventsite_flags) else 0 
                                           for index, row in pois_filtered.iterrows()]

    # calculate the centroids and drop irrelevant columns
    pois_filtered["geometry"] = pois_filtered.geometry.to_crs(epsg=3310).centroid.to_crs(epsg=4326)

    pois_filtered = pois_filtered.drop(columns=["landuse", "building"]).loc[(pois_filtered.commercial == 1) | (pois_filtered.residential == 1) | 
                                                                            (pois_filtered.transportation == 1) | (pois_filtered.schools == 1) | (pois_filtered.eventsites == 1)]
    return pois_filtered


def merge_pois_with_street_network(street_network, pois, radius=500):
    """
    Define circles with radius 500m around the POIs and merge those circles with the street network via gpd.sjoin
    Runtime: 20mins
    """
    
    print("Defining circles around POIs...")
    pois = define_circles_around_points(geodataframe=pois, radius=radius)
    
    print("Merging circles with street network...")
    streets_pois_sjoin_1 = gpd.sjoin(street_network, pois[pois.commercial == 1][["geometry", "commercial"]], how="left")
    streets_pois_grouped_1 = streets_pois_sjoin_1.groupby(by=["u","v"]).agg({"commercial": "sum"})
    streets_with_pois_1 = pd.merge(street_network[['osmid', 'oneway', 'lanes', 'ref', 'highway', 'maxspeed', 'length', 'geometry', 'from', 'to', 'bridge', 
                                                          'name', 'tunnel','junction', 'width', 'access']], 
                                   streets_pois_grouped_1, on=["u", "v"], how="inner")

    streets_pois_sjoin_2 = gpd.sjoin(streets_with_pois_1, pois[pois.residential == 1][["geometry", "residential"]], how="left")
    streets_pois_grouped_2 = streets_pois_sjoin_2.groupby(by=["u","v"]).agg({"residential": "sum"})
    streets_with_pois_2 = pd.merge(streets_with_pois_1[['osmid', 'oneway', 'lanes', 'ref', 'highway', 'maxspeed', 'length', 'geometry', 'from', 'to', 'bridge', 
                                                          'name', 'tunnel','junction', 'width', 'access', 'commercial']], 
                                   streets_pois_grouped_2, on=["u", "v"], how="inner")

    streets_pois_sjoin_3 = gpd.sjoin(streets_with_pois_2, pois[pois.transportation == 1][["geometry", "transportation"]], how="left")
    streets_pois_grouped_3 = streets_pois_sjoin_3.groupby(by=["u","v"]).agg({"transportation": "sum"})
    streets_with_pois_3 = pd.merge(streets_with_pois_2[['osmid', 'oneway', 'lanes', 'ref', 'highway', 'maxspeed', 'length', 'geometry', 'from', 'to', 'bridge', 
                                                          'name', 'tunnel','junction', 'width', 'access', 'commercial', 'residential']], 
                                   streets_pois_grouped_3, on=["u", "v"], how="inner")

    streets_pois_sjoin_4 = gpd.sjoin(streets_with_pois_3, pois[pois.schools == 1][["geometry", "schools"]], how="left")
    streets_pois_grouped_4 = streets_pois_sjoin_4.groupby(by=["u","v"]).agg({"schools": "sum"})
    streets_with_pois_4 = pd.merge(streets_with_pois_3[['osmid', 'oneway', 'lanes', 'ref', 'highway', 'maxspeed', 'length', 'geometry', 'from', 'to', 'bridge', 
                                                          'name', 'tunnel','junction', 'width', 'access', 'commercial', 'residential', 'transportation']], 
                                   streets_pois_grouped_4, on=["u", "v"], how="inner")

    streets_with_eventsites = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")

    streets_pois_sjoin_5 = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")
    streets_pois_grouped_5 = streets_pois_sjoin_5.groupby(by=["u","v"]).agg({"eventsites": "sum"})
    streets_with_pois_5 = pd.merge(streets_with_pois_4[['osmid', 'oneway', 'lanes', 'ref', 'highway', 'maxspeed', 'length', 'geometry', 'from', 'to', 'bridge', 
                                                          'name', 'tunnel','junction', 'width', 'access', 'commercial', 'residential', 'transportation', 'schools']], 
                                   streets_pois_grouped_5, on=["u", "v"], how="inner")
    
    return streets_with_pois_5


def merge_pois_with_final_result(street_network, pois, radius=500):
    """
    Define circles with radius 500m around the POIs and merge those circles with the the final dataframe which was generated before via gpd.sjoin
    Runtime: 20mins
    """
    
    print("Defining circles around POIs...")
    pois = define_circles_around_points(geodataframe=pois, radius=radius)
    
    print("Merging circles with street network...")
    streets_pois_sjoin_1 = gpd.sjoin(street_network, pois[pois.commercial == 1][["geometry", "commercial"]], how="left")
    streets_pois_grouped_1 = streets_pois_sjoin_1.groupby(by=["street_id"]).agg({"commercial": "sum"})
    streets_with_pois_1 = pd.merge(street_network, 
                                   streets_pois_grouped_1, on=["street_id"], how="inner")

    streets_pois_sjoin_2 = gpd.sjoin(streets_with_pois_1, pois[pois.residential == 1][["geometry", "residential"]], how="left")
    streets_pois_grouped_2 = streets_pois_sjoin_2.groupby(by=["street_id"]).agg({"residential": "sum"})
    streets_with_pois_2 = pd.merge(streets_with_pois_1,
                                   streets_pois_grouped_2, on=["street_id"], how="inner")

    streets_pois_sjoin_3 = gpd.sjoin(streets_with_pois_2, pois[pois.transportation == 1][["geometry", "transportation"]], how="left")
    streets_pois_grouped_3 = streets_pois_sjoin_3.groupby(by=["street_id"]).agg({"transportation": "sum"})
    streets_with_pois_3 = pd.merge(streets_with_pois_2, 
                                   streets_pois_grouped_3, on=["street_id"], how="inner")

    streets_pois_sjoin_4 = gpd.sjoin(streets_with_pois_3, pois[pois.schools == 1][["geometry", "schools"]], how="left")
    streets_pois_grouped_4 = streets_pois_sjoin_4.groupby(by=["street_id"]).agg({"schools": "sum"})
    streets_with_pois_4 = pd.merge(streets_with_pois_3, 
                                   streets_pois_grouped_4, on=["street_id"], how="inner")

    streets_with_eventsites = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")

    streets_pois_sjoin_5 = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")
    streets_pois_grouped_5 = streets_pois_sjoin_5.groupby(by=["street_id"]).agg({"eventsites": "sum"})
    streets_with_pois_5 = pd.merge(streets_with_pois_4, 
                                   streets_pois_grouped_5, on=["street_id"], how="inner")
    
    return streets_with_pois_5




def merge_pois_with_street_network_melbourne(street_network, pois, radius=500, merge_key="rd_seg_id"):
    """
    Define circles with radius 500m around the POIs and merge those circles with the street network via gpd.sjoin
    Runtime: 20mins
    """
    
    print("Defining circles around POIs...")
    pois = define_circles_around_points(geodataframe=pois, radius=radius)
    
    print("Merging circles with street network...")
    streets_pois_sjoin_1 = gpd.sjoin(street_network, pois[pois.commercial == 1][["geometry", "commercial"]], how="left")
    streets_pois_grouped_1 = streets_pois_sjoin_1.groupby(by=merge_key).agg({"commercial": "sum"})
    streets_with_pois_1 = pd.merge(street_network, streets_pois_grouped_1, on=merge_key, how="inner")

    streets_pois_sjoin_2 = gpd.sjoin(streets_with_pois_1, pois[pois.residential == 1][["geometry", "residential"]], how="left")
    streets_pois_grouped_2 = streets_pois_sjoin_2.groupby(by=merge_key).agg({"residential": "sum"})
    streets_with_pois_2 = pd.merge(streets_with_pois_1, streets_pois_grouped_2, on=merge_key, how="inner")

    streets_pois_sjoin_3 = gpd.sjoin(streets_with_pois_2, pois[pois.transportation == 1][["geometry", "transportation"]], how="left")
    streets_pois_grouped_3 = streets_pois_sjoin_3.groupby(by=merge_key).agg({"transportation": "sum"})
    streets_with_pois_3 = pd.merge(streets_with_pois_2, streets_pois_grouped_3, on=merge_key, how="inner")

    streets_pois_sjoin_4 = gpd.sjoin(streets_with_pois_3, pois[pois.schools == 1][["geometry", "schools"]], how="left")
    streets_pois_grouped_4 = streets_pois_sjoin_4.groupby(by=merge_key).agg({"schools": "sum"})
    streets_with_pois_4 = pd.merge(streets_with_pois_3, streets_pois_grouped_4, on=merge_key, how="inner")

    streets_with_eventsites = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")

    streets_pois_sjoin_5 = gpd.sjoin(streets_with_pois_4, pois[pois.eventsites == 1][["geometry", "eventsites"]], how="left")
    streets_pois_grouped_5 = streets_pois_sjoin_5.groupby(by=merge_key).agg({"eventsites": "sum"})
    streets_with_pois_5 = pd.merge(streets_with_pois_4, streets_pois_grouped_5, on=merge_key, how="inner")
    
    return streets_with_pois_5