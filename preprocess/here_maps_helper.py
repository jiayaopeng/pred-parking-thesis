import requests
import pandas as pd
import boto3
import geopandas as gpd
from geolocation_helper import define_circles_around_points
import xmltodict
import json
import time

def get_ssm_param(param_name, decrypted_param=False):
    ssm_client = boto3.client('ssm')
    param_dict = ssm_client.get_parameter(Name=param_name, WithDecryption=decrypted_param)
    val = param_dict['Parameter']['Value']
    return val
    
def make_discover_request(box, query):
    '''
    Make a request to the here discovery 
    '''
    discover_api_key = get_ssm_param('here_discovery_api_key', True)
    #discover_api_key = 'N5hljILMSlVzdIZ1FDn_gLZLyNTb4GzDUFSrEkAA8vU'
    params= (
    ('apiKey', discover_api_key),
    ('in', 'bbox:'+','.join(box) ),
    ('q', query),
    ('limit', '100')
    )
    resp = requests.get("https://discover.search.hereapi.com/v1/discover", params=params )
    return resp

def make_on_street_parking_request(box):
    '''
    returns all on-street parking segments deined by here in the specified bounding box
    '''
    onstreet_api_key = get_ssm_param('here_onstreet_parking_api_key', True)
    #onstreet_api_key = '3UUXgfq34chpny3MRJ6yYkuBuUxX57btvZMRUL2aWJw'
    params = (
    ('bbox', ','.join([str(box[i]) for i in [1,0,3,2]])),
    ('apiKey' , onstreet_api_key),
        ('geometryType',  'tpegOpenLR') #'segmentAnchor',
    )
    headers = {
    'Accept-Encoding': 'gzip, deflate, sdch, br',
    }
    response = requests.get('https://on-street-parking-dev.api-gateway.sit.ls.hereapi.com/parking/segments', headers=headers, params=params)
    return response

def make_off_street_parking_request(lat, lng, radius):
    '''
    returns all off-street-parking facilities in teh radius around the specified point
    '''
    offstreet_api_key = get_ssm_param('here_offstreet_parking_api_key', True)
    #offstreet_api_key = '3UUXgfq34chpny3MRJ6yYkuBuUxX57btvZMRUL2aWJw'
    params = (
    ('prox', ",".join([str(lat), str(lng), str(radius)])),
    ('apiKey' , offstreet_api_key)
    )

    response = requests.get('https://off-street-parking-ix-dev.api-gateway.sit.ls.hereapi.com/parking/search',  params=params)
    return response


def get_buildings_in_bounding_box(city, box, building, save=False):
    '''
    Makes a request to the here discovery endpoint askin for all specified buildings in the bounding box
    '''
    response = make_discover_request(box, building)
    if response.status_code == 429:
        print('the status is 429')
        time.sleep(int(response.headers["Retry-After"]))
        return get_buildings_in_bounding_box(city, box, building, save)
    elif response.status_code != 200:
        print(response.content)
        print(f'failed for box {box} and buiding{building}')
        df = pd.DataFrame()
        print("empty dataframe has been created")
    else:
        df = pd.DataFrame(response.json()['items'])
    if save: 
        df.to_csv(f'{city}_{building}.csv')
    return df
    
def get_buildings_within_radius(location_df, radius_in_meter, building_list, city_name):
    '''
    returns the number of buildings per building type in building list (e.g. [restaurants, supermarkets]) 
    within the specified radius around a location
    '''
    location_df.geometry = location_df.geometry.map(lambda x: x.centroid)
    street_geoms = define_circles_around_points(geodataframe=location_df, radius=radius_in_meter)
    for building in building_list:
        #Get a list of all buidlings of the specific type within the specified radius in the inner list comprehension. 
        #Only take the lenght of the list (the number of the specific buildings in the nieghbourhood in the outer list comprehension as result)
        street_geoms[building] = [len(res) for res in [get_buildings_in_bounding_box(city_name, [str(x) for x in geom.bounds], building) for geom in street_geoms.geometry]]
    return street_geoms

def add_neighbourhood_info_here(data, radius_in_meter, building_list, city_name, location_identifier = 'street_id'):
    #Create a dataframe with information per street
    street_geoms = data[[location_identifier, 'geometry']].drop_duplicates(subset=location_identifier)
    street_geoms['location_geom'] = street_geoms.geometry
    street_geoms = get_buildings_within_radius( street_geoms, radius_in_meter, building_list, city_name)
    
    data = data.merge(street_geoms[building_list + [location_identifier]] , on=location_identifier, suffixes=('','_here'))
    return data
    
    
    
def get_parking_segments_in_bounding_box(box):
    resp = make_on_street_parking_request(box)
    if resp.status_code==200:
        result = resp.json()['parkingSegments'] 
    else:
        result =[]
    return result

def add_static_parking_info_here(data, location_identifier='street_id'):
    '''
    Each row in the data df contains an observation at a given location and a given time 
    This function adds the current capacity per row (capacity at the given time for a given street) to the dataframe 
    This is necessary because sometimes a street consists of several parking locations with different opening hours 
    (e.g. left side of the street parking is allowed from 10-15 while right side has opening hours 10-18)
    '''
    street_geoms = data[[location_identifier, 'geometry']].drop_duplicates(subset=location_identifier)
    #Build a small bounding box around each street to query all parking segemnts from here in that box
    street_geoms['street_box'] = gpd.GeoDataFrame(geometry=street_geoms.geometry).set_crs(epsg=4326).to_crs(epsg=3395).buffer(20, cap_style=3).to_crs(epsg=4326)
    # Get all parking segements in that street
    street_geoms['static_parking'] = [get_parking_segments_in_bounding_box(geom.bounds) for geom in street_geoms.street_box]
    #Get the maximum capacity ignoring opening hours
    street_geoms['capacity'] = street_geoms.static_parking.map(lambda park_list: sum([x.get('capacity', 0) for x in park_list]))
    #Get the hourly capacity based on opening hours 
    street_geoms['hourly_capacity'] = street_geoms.static_parking.map(lambda park_list: get_hourly_capacity_lookup_from_response(park_list))
    #Merge capacity infomration to the data 
    data_with_capa = data.merge(street_geoms[[location_identifier, "capacity", "hourly_capacity"]], on = location_identifier, how='left', suffixes=('','_here'))
    data_with_capa['current_capacity'] = data_with_capa.apply(lambda x: x.hourly_capacity[str(x.hour)], axis=1)
    return data_with_capa
        

def get_hourly_capacity_lookup_from_response(parking_info):
    '''
    Aggregates the capacity of all parking locations in that bounding box on an hourly level
    One list can contain different segemnts with different opening hours, so the capacity of the segemnt changes over the day
    return a dicotnary with hour of the day as key and capacity as value
    '''
    start_end = [info.get('priceSchema',{}).get('prices',[{}])[0].get('times', [{}])[0].get('timeRange',{}).values() for info in parking_info]
    if sum([len(x) for x in start_end])==0:
        return {str(hour): 0 for hour in range(24)}
    capacity = [info.get('capacity',0) for info in parking_info]
    cap_df = pd.DataFrame(start_end, columns=['start', 'end'])
    cap_df['capacity'] = capacity
    cap_df = cap_df[(cap_df.capacity>0) & (cap_df.end.notna())]
    cap_df.start = cap_df.start.map(lambda x: x.split(':')[0]).astype(int)
    cap_df.end = cap_df.end.map(lambda x: x.split(':')[0]).astype(int)
    capacity_lookup = {str(hour) : cap_df[(cap_df.start<=hour) & (cap_df.end>=hour)].capacity.sum() for hour in range(24)}
    return capacity_lookup

def get_off_street_parking_faciliy_from_response(api_response):
    if api_response.status_code == 200:
        json_response =  json.loads(json.dumps(xmltodict.parse(api_response.content)))
        facility_dict = json_response['parking-offstreet:parkingFacilitiesResult']['facilities']
        if type( facility_dict)==dict:
            facility_list = facility_dict['facility']
        else:
            facility_list = []
    else:
        print(f'status code {api_response.status_code}')
        facility_list = []
    return facility_list
    
def get_off_street_parking_capacity(facility_list):
    if type(facility_list)==list:
        num_off_street_parking = len(facility_list)
        off_street_capa = sum([float(facility['facilityAvailability'].get('spacesTotal',0)) for facility in facility_list])
    elif type(facility_list) == dict:
        num_off_street_parking = 1
        off_street_capa = facility_list['facilityAvailability'].get('spacesTotal',0)
    else:
        num_off_street_parking = 0
        off_street_capa = 0
    return [num_off_street_parking, off_street_capa]
    
    
def compute_off_street_features(location_geom, radius_in_meter):
    api_response = make_off_street_parking_request(location_geom.centroid.y, location_geom.centroid.x, radius_in_meter)
    facility_list = get_off_street_parking_faciliy_from_response(api_response)
    capa = get_off_street_parking_capacity(facility_list)
    #TODO compute price information here 
    return capa

def add_off_street_parking_here(data, radius, location_identifier='street_id'):
    street_geoms = data[[location_identifier, 'geometry']].drop_duplicates(subset=location_identifier)
    street_geoms[['num_off_street_parking', 'off_street_capa']] = [compute_off_street_features(location_geom=geom, radius_in_meter = radius) for geom in street_geoms.geometry]
    street_geoms[['num_off_street_parking', 'off_street_capa']] = street_geoms[['num_off_street_parking', 'off_street_capa']].astype(int)
    data_with_offstreet = data.merge(street_geoms[[location_identifier, 'num_off_street_parking', 'off_street_capa']], on = location_identifier, how='left', suffixes=('','_here'))
    return data_with_offstreet
    