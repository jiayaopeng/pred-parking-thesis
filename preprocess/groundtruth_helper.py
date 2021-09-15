import numpy as np
import pandas as pd
from geolocation_helper import merge_df_on_nearest_geometries



def compute_gt_labels(pred_geom, groundtruth_data, pred_geom_id='street_id', spatial_gran_col='length', spatial_gran=200, temporal_gran=5):
    #Assign each PaybyPhone location to the nearest prediction geometry
    groundtruth_assign = merge_df_on_nearest_geometries(groundtruth_data, pred_geom, gdfB_cols=[pred_geom_id])
    #Compute all PBP locations that belong to a given prediction geometrie
    all_locations_to_geom = groundtruth_assign.groupby(pred_geom_id).apply(lambda x: x.advLocationId.unique()).reset_index(name='locations_to_street')
    groundtruth_assign = groundtruth_assign.merge(all_locations_to_geom, on=pred_geom_id)
    groundtruth_assign = groundtruth_assign.merge(pred_geom[[pred_geom_id,'highway', 'maxspeed', 'length']].astype(str).drop_duplicates(), on=pred_geom_id)

    groundtruth_assign_timegroup = groundtruth_assign.groupby(pred_geom_id).apply(lambda x: define_time_group(x, temporal_gran)).reset_index(drop=True)
    groundtruth_labels = groundtruth_assign_timegroup.groupby([pred_geom_id, 'observation_interval_start']).apply(lambda x: check_street_level_occupancy_labels(x, temporal_gran, spatial_gran))
    groundtruth_labels.dropna(inplace=True)
    return groundtruth_labels
    

def define_time_group(df, temporal_gran):
    '''
    To check the availability of a location we have to find all observations of PayByPhone street segments that abelong to that location and are close togehter timewise 
    We define observations that belong together (same street, similar time) by the same observation_interval_start identifier. 
    This identifier is set as the timestamp of the earliest observation of the group
    '''
    df = df.sort_values('time_stamp')
    df['observation_interval_start'] = df.time_stamp
    #When we observe a time gap of ten minutes between two observations start a new group
    new_group_indexes = df.time_stamp.diff() >= pd.Timedelta(minutes=temporal_gran)
    new_group_indexes.iloc[0] = True
    # Replace interval start with nans if no new group is started 
    df['observation_interval_start'].where(new_group_indexes, inplace=True)
    #Fill nans with previuosly observed value (start time of group)

    df.observation_interval_start.fillna(method='ffill', inplace=True)
    
    return df

def check_street_level_occupancy_labels(df, temporal_gran, spatial_gran):
    '''
    We define a stret to be available if there is at least one free parking per 200 meter length within 5 minute intervals
    
    This functions assumes it is called for one location-5minute interval (from a groupby operation)
    Assume we have a street consisting of two PayByPhone street segments (street_segment_1 is observed at time t while segment 2 is observed at time t+1)
    When the distance between t and t+1 is smaller than the temporal granularity (5mins) we sum all the observations and check for availability
    
    '''
    
    
    result = pd.DataFrame({ 'availability': [np.nan]})
    if (max(df['time_stamp']) - min(df['time_stamp'])) <= pd.Timedelta(minutes=temporal_gran):
        if set(df['advLocationId'].values) == set(df['locations_to_street'].iloc[0]):
            street_agg = df[['total_vehicle_count', 'parking_spaces']].sum()
            street_avail = (street_agg.parking_spaces - street_agg.total_vehicle_count)>= np.ceil(float(df['length'].iloc[0])/spatial_gran)
            result = pd.DataFrame({'availability': [street_avail]})

    return result