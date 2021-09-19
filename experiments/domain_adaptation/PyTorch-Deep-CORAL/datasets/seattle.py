import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import encode.encode as lsh


def generate_train_data(data, features, label, val_size, test_size):
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[label].astype("int"), test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size / (1 - test_size))
    return {'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test': (x_test, y_test)}


def create_data(data, source_areas, target_areas, features, label, val_size, test_size):
    source_data = data[[x in source_areas for x in data.study_area]]
    target_data = data[[x in target_areas for x in data.study_area]]
    ##################
    # Split the data #
    ##################
    # train - val (test 80, val 20)
    data_split_target = generate_train_data(target_data, features=features, label=label, val_size=val_size,
                                        test_size=0.001)
    # whole train
    data_split_source = generate_train_data(source_data, features=features, label=label, val_size=0.0001, test_size=0.0001)
    # not split target at all
    ######################
    # Encode categorical #
    ######################
    # encoder target train, and generate encoder
    encoded_target_train, target_encoder = lsh.encode_categorical(
        encoder='target_encoder',
        col_encoded=['highway', 'hour', 'weekday'],
        feature=data_split_target['train'][0],
        target=data_split_target['train'][1]
    )
    # use train encoder to encode test and val
    encoded_target_val = target_encoder.transform(X=data_split_target['val'][0],y=data_split_target['val'][1])
    encoded_target_test = target_encoder.transform(X=data_split_target['test'][0], y=data_split_target['test'][1])

    # encode source
    encoded_source_train, source_encoder = lsh.encode_categorical(
        encoder='target_encoder',
        col_encoded=['highway', 'hour', 'weekday'],
        feature=data_split_source['train'][0],
        target=data_split_source['train'][1]
    )
    # use train encoder to encode test and val
    encoded_source_val = source_encoder.transform(X=data_split_source['val'][0], y=data_split_source['val'][1])
    encoded_source_test = source_encoder.transform(X=data_split_source['test'][0], y=data_split_source['test'][1])

    ################
    # Scale x-data #
    ################
    # for source and target data, initialize the scaler for train and source data
    target_scaler = StandardScaler()
    source_scaler = StandardScaler()
    # for target, initialize with train data by fit_transform, and use it to valid and test data
    x_train_target = target_scaler.fit_transform(X=encoded_target_train)
    x_val_target = target_scaler.transform(encoded_target_val)
    x_test_target = target_scaler.transform(encoded_target_test)

    y_train_target = data_split_target['train'][1].values
    y_val_target = data_split_target['val'][1].values
    y_test_target = data_split_target['test'][1].values

    # for source,  initialize with train data by fit_transform, and use it to valid and test data
    x_train_source = source_scaler.fit_transform(X=encoded_source_train)
    x_val_source = source_scaler.transform(encoded_source_val)
    x_test_source = source_scaler.transform(encoded_source_test)

    y_train_source = data_split_source['train'][1].values
    y_val_source = data_split_source['val'][1].values
    y_test_source = data_split_source['test'][1].values

    return {'target':
                {'train': (x_train_target, y_train_target),
                 'val': (x_val_target, y_val_target),
                 'test': (x_test_target, y_test_target)},
            'source':
                {'train': (x_train_source, y_train_source),
                 'val': (x_val_source, y_val_source),
                 'test': (x_test_source, y_test_source)}
            }


def load_seattle(target_areas, source_areas, include_pbp, bucket, sm_mode):
    if not sm_mode:
        train_data_with_trans = pd.read_csv('train_data_with_trans_100_with_transaction.csv',
                                            index_col=0)
    else:
        train_data_with_trans = pd.read_csv('/opt/ml/input/data/seattle/train_data_with_trans_100_with_transaction.csv',
                                            index_col=0) # read the data to the tuner

    train_data_with_trans.observation_interval_start = pd.to_datetime(train_data_with_trans.observation_interval_start)

    # filter out only 9 areas as for baseline and clustering
    selected_areas = [
        'Greenlake',
        'South Lake Union',
        'Commercial Core',
        'Pike-Pine',
        'Uptown',
        'Ballard',
        'First Hill',
        'Chinatown/ID',
        'Pioneer Square'
    ]
    train_data_with_trans = train_data_with_trans[
        train_data_with_trans["study_area"].isin(selected_areas)
    ]

    # select the same 21+1 features as for baseline and clustering
    features = ['length', 'current_capacity', 'tempC', 'windspeedKmph', 'precipMM', # 5
                'highway', 'hour', 'weekday', 'commercial_100', 'residential_100', # 5
                'transportation_100', 'schools_100', 'eventsites_100', # 3
                'restaurant_here_100', 'shopping_here_100', 'office_here_100', # 3
                'supermarket_here_100', 'transportation_here_100', 'schools_here_100', # 3
                'num_off_street_parking_100', 'off_street_capa_100'] # 2

    if include_pbp:
        features = features + ['ongoing_trans']

    label = 'availability'

    experiment_setup = create_data(train_data_with_trans, source_areas, target_areas, features, label, val_size=0.25,
                                   test_size=0.02)

    return experiment_setup