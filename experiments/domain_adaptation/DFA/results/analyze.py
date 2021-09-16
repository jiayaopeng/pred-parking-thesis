import pandas as pd

import results.visualize as rv


def get_mean_of_mean(df_means):
    """
    This function get the mean of means by averaging over the mean for all area combinations
    """
    
    ls_settings = list(df_means.index.unique(level='setting'))

    mean_of_mean = {} # 1) for each area, avg all experiments 2) the avg across different areas
    
    for setting in ls_settings:
        mean_of_mean[setting] = df_means.loc[(slice(None), setting), :].mean(axis=0).to_dict()

    mean_of_mean = pd.DataFrame(mean_of_mean).rename(
        index = {
            'val_matthew':'avg_val_matthew',
            'test_matthew': 'avg_test_matthew'
        }
    ).T
    
    return mean_of_mean


def corr_area_size_matthew(df_mean, highest_val_mean):
    """
    Correlation between number of data points in one area and its matthew
    """
    # get number of data points per area
    dataset = pd.read_csv(
        's3://bucket-vwfs-pred-park-global-model-serving-dev/input/open_data/seattle/train_data_with_trans_100_with_transaction.csv',
        index_col=0
    )
    dataset = dataset.groupby('study_area').filter(lambda x: len(x) > 250)
    area_data_size = pd.DataFrame(dataset.groupby('study_area').size(), columns=['data_size'])

    df_matthew_size = df_mean.merge(area_data_size, left_index=True, right_index=True)
    print("")
    
    # plot
    rv.plot_corr_matthew_datasize(df_matthew_size, highest_val_mean)
    
    return df_mean


