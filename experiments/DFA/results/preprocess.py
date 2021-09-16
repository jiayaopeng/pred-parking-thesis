import boto3
import io
import tarfile
import json
import numpy as np
import pandas as pd


def get_results(name_dict):
    """
    This function is used to get result of the final training job. 
    This final job was trained by specific number of experiments, and 
    used the the best hyperparameter combination after the hyperparameter tuning.
    The results is the result which we logged we train the model
    input:
        a dictionary of training job names
    Output:
        a nested dictionary where 
            first level key: training job name
            second level key: test metric mainly (but we also log validation mattew)
            third level key: different area combinations
            value: we calculate the metric for each experiment
    """
    s3 = boto3.resource('s3') 
    s3_client= boto3.client('s3')
    res = {}
    for job in name_dict['estimators']:
        bucket_name = job.split('s3://')[1].split('/')[0]
        bucket = s3.Bucket(bucket_name) 
        for obj in bucket.objects.filter(Prefix=job.split(bucket_name+'/')[1]):
            if obj.key.split('/')[-1]!='model.tar.gz':
                continue
            s3_object = s3_client.get_object(Bucket=bucket_name, 
                                 Key=obj.key)
            wholefile = s3_object['Body'].read()
            fileobj = io.BytesIO(wholefile)
            tar = tarfile.open(fileobj=fileobj)
            f = tar.extractfile(tar.getmembers()[0])
            key = '_'.join(obj.key.split('/')[3:6])
            f = json.load(f)
            
        # save the dictionary of dictionary
        # this if condiction has been added as for one final run we ran 50 experiments, but we 20 experiment 
        # can already give us meaningful statistics
        res[key] = {}
        metrics = ['auc', 'matthew', 'f1', 'fbeta', 'accuracy', 'precision', 'recall', 'val_matthew']
        # loop through the metrics
        for metric in metrics: 
            # only select the first 20 experiments
            res[key][metric] = {}
            for area, data in f.get(metric).items():
                res[key][metric][area] = data[:20]

    return res


def add_exp_data(exp_name, exp_data, output, get_mean):
    """
    This function does two main things: 
        1)swap the keys 
        2)for each area, calculate the mean of the metric over multiple experiments
    Input:
        exp_name: job_name of the experiment
        exp_data: data
        output: dictionary to be initialized 
    Output:
        A nested dictionary:
            first level key: different area combinations
            second level key: name of the training job
            third level key: metric
            value: we average the metric for each experiment
            
    """
    for metric, area_values in list(exp_data.items()):
        for area, values in list(area_values.items()):
            if area not in output:
                output[area] = {}
            if exp_name not in output[area]:
                output[area][exp_name] = {}
            if not output[area][exp_name]:
                output[area][exp_name] = {}
            if get_mean:
                output[area][exp_name][metric] = round(np.mean(values), 2)
            else:
                output[area][exp_name][metric] = values

            
def get_metric_df(s3_res, get_mean=False):
    """
    This function could be used to 
        1)get the mean of all metrics from the queried data from s3 training job
        or 
        2)does not return mean, but return a list of values for all experiment in the cell
    """
    # analyse the data, create nested dictionary to hold mean
    output = {}

    # get the average of the experiments of matthews
    for exp_name, exp_data  in s3_res.items():
        add_exp_data(exp_name, exp_data, output, get_mean)

    # create the multi index dataframe
    # Reference: https://stackoverflow.com/questions/47416113/how-to-build-a-multiindex-pandas-dataframe-from-a-nested-dictionary-with-lists
    df = pd.DataFrame.from_dict({(i,j): output[i][j]
                            for i in output.keys()
                            for j in output[i].keys()},
                            orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    
    return df

    
def select_matthew_rename(df, name_map):
    """
    This function select only the matthew in the metric and rename the area combination with only the targe
    """
   
    df_matthew = df[['val_matthew', 'matthew']]

    # rename the index of matthew with only the name of target area and matthew to test matthew
    df_matthew = df_matthew.rename(
        columns={ 
          'matthew': 'test_matthew'
        },
        index=name_map
    )
    
    # give the multi index names
    df_matthew=df_matthew.rename_axis(['target area','setting'])
    
    
    return df_matthew


def unnesting(df, explode):
    """
    Reference: https://stackoverflow.com/questions/56499336/how-to-convert-nested-dict-with-lists-as-values-to-pandas-dataframe
    """
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


def preprocess_for_boxplot(df, test_matthew_only=False):
    if not test_matthew_only:
        # unnesting
        unnested_data = unnesting(df, ['val_matthew', 'test_matthew']).reset_index()
        # melt
        melted_data= pd.melt(unnested_data, id_vars=['target area'], value_vars=['val_matthew', 'test_matthew'])
        # rename
        renamed_data =melted_data.rename(
            columns={
                'index': 'target_area',
                'variable':'metric',
                'value': 'matthew_score'
            }
        )
        return renamed_data
    
    else: # if we only want to plot test box plot
        df_test_matthew = df.drop(['val_matthew'], axis = 1)
        # unest
        unnested_data = unnesting(df_test_matthew , ['test_matthew']).reset_index()
        # melt
        melted_data= pd.melt(unnested_data, id_vars=['target area'], value_vars=['test_matthew'])
        # rename
        renamed_data =melted_data.rename(
            columns={
                'index': 'target_area',
                'variable':'metric',
                'value': 'matthew_score'
            }
        )
        
        return renamed_data