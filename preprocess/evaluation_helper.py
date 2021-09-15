from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, fbeta_score
from model_training_helper import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
import itertools

#Set plot style
sns.set(rc={
 "axes.axisbelow": False,
 "axes.edgecolor": "lightgrey",
 "axes.facecolor": "None",
 "axes.grid": False,
 "axes.labelcolor": "dimgrey",
 "axes.spines.right": False,
 "axes.spines.top": False,
 "figure.facecolor": "white",
 "lines.solid_capstyle": "round",
 "patch.edgecolor": "w",
 "patch.force_edgecolor": True,
 "text.color": "dimgrey",
 "xtick.bottom": False,
 "xtick.color": "dimgrey",
 "xtick.direction": "out",
 "xtick.top": False,
 "ytick.color": "dimgrey",
 "ytick.direction": "out",
 "ytick.left": False,
 "ytick.right": False})
sns.set_context("notebook", rc={"font.size":16,
                                "axes.titlesize":20,
                                "axes.labelsize":18})


def generate_train_data(data, features, label, val_size, test_size):
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[label].astype("int"), test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size= val_size/(1-test_size))
    return {'train' : (x_train, y_train),
           'val': (x_val, y_val),
           'test': (x_test, y_test)}

def generate_train_data_disjunct_locations(data, features, label, val_size, test_size, location_identifier='street_id'):
    '''
    generates a train-test split where train and test have completly disjunct locations 
    '''
    #sample locations
    locations = data[location_identifier].unique()
    test_locations = np.random.choice(locations, int(test_size*len(locations)), replace=False)
    val_locations = np.random.choice([loc for loc in locations if loc not in test_locations], int(val_size*len(locations)), replace=False)
    #query data
    train_data = data[[loc not in (test_locations.tolist() + val_locations.tolist()) for loc in data[location_identifier]]]
    val_data = data[[loc in val_locations for loc in data[location_identifier]]]
    test_data = data[[loc in test_locations for loc in data[location_identifier]]]
    #SPlit x,y 
    x_train, y_train = train_data[features], train_data[label].astype(int)
    x_val, y_val = val_data[features], val_data[label].astype(int)
    x_test, y_test = test_data[features], test_data[label].astype(int)
    return {'train' : (x_train, y_train),
           'val': (x_val, y_val),
           'test': (x_test, y_test)}
    
    


def evaluate_performance(pred, probas, y_test, print_res = False):
    fbeta = fbeta_score(y_true=y_test, y_pred=pred, beta=0.33)
    mathew_corr = matthews_corrcoef(y_true=y_test, y_pred=pred)
    accuracy = accuracy_score(y_pred=pred,y_true=y_test)
    if print_res:
        print(f'recall {recall_score(y_pred=pred,y_true=y_test)}')
        print(f'precision {precision_score(y_pred=pred,y_true=y_test)}')
        print(f'accuracy {accuracy}')
        print(f'auc-proba {roc_auc_score(y_score=probas, y_true=y_test)}')
        print(f'auc-thres {roc_auc_score(y_score=pred, y_true=y_test)}')
        print(f'F1-Score: {f1_score(y_true=y_test, y_pred=pred)}')
        print(f'FBeta(0.33)-Score: {fbeta}')
        print(f'Matthew Correlation: {mathew_corr}')
    return {'F-Beta': fbeta, 'matthew': mathew_corr, 'accuracy': accuracy}

def plot_feature_group_performance(performance_dict):
    res_df = pd.concat([pd.DataFrame(performance, index =[group]) for group,performance in performance_dict.items()])
    plot_df = pd.DataFrame(res_df.unstack()).reset_index()
    plot_df.columns= ['metric', 'feature_source', 'score']
    plt.figure()
    perf_plot = sns.barplot(data=plot_df, x='metric', y='score', hue='feature_source')
    plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    fig = perf_plot.get_figure()
    fig.savefig('feature_groups.png', bbox_inches='tight')
    #fig.clf()
    

def plot_feature_importance(feature_description_dict, model):
    feature_importance = model.get_feature_importance()
    # Read description dict
    feature_groups, all_features, cat_features = read_feature_description_dict(feature_description_dict)
    numerical_feat = [feat for feat in all_features if feat not in cat_features]
    # Create and plot dataframe with feature importances 
    feature_importance_df = pd.DataFrame({'feature_importance': feature_importance, 'feature_name': cat_features + numerical_feat})
    plt.figure(figsize=(10, 20))
    feat_importance_plot = sns.barplot(data=feature_importance_df.sort_values('feature_importance', ascending=False), y='feature_name', x='feature_importance', orient='h')
    fig = feat_importance_plot.get_figure()
    fig.savefig('feature_importance.png', bbox_inches='tight') 

    
def read_feature_description_dict(feature_description_dict):
    if not 'cat_features' in list(feature_description_dict.keys()):
        raise Exception("must provide cat_features key")
    
    feature_groups = [feature_group for feature_group in feature_description_dict.keys() if feature_group!='cat_features']
    all_features = [feature_description_dict[group] for group in feature_groups]
    all_features = list(set([item for sublist in all_features for item in sublist]))
    cat_features = feature_description_dict['cat_features']
    
    return feature_groups, all_features, cat_features

    
    
def compare_feature_combinations(data, n, feature_description_dict, label_name, val_size, test_size, disjunct_locations=False, perform_t_test=False):
    '''
    Input a dictonary with different feature groups 
    keys: group names, values. list of feature names
    example
    feature_description_dict = {
    'time_features': ['hour', 'weekday'],
    'map_features' : ['restaurants', 'event_sides'],
    'cat_features': ['hour', 'weekday']
    }
    n is the number of different train-test splits. The final result wis the average performance among all splits
    If disjunct locations = true use different locations for training and testing
    '''


    feature_groups, all_features, cat_features = read_feature_description_dict(feature_description_dict)
    
    #Get a sample of the evaluation metric sto init all of them
    # The [0,1] is just a sample of a model prediction and target (perfect model) since we are onlyiintereted in the metrics but not in the values
    performance_metrics = evaluate_performance([0,1], [0,1], [0,1]).keys()
    # Initialaize the performance dict which is a nested dictonary: For each feature group we have a dict specifiying all relevant performance metrics and their value
    perf_dict = {group : {metric: [] for metric in performance_metrics} for group in feature_groups}
    
    for i in range(n):
        
        if disjunct_locations:
            data_dict = generate_train_data_disjunct_locations(data, all_features, label_name, val_size, test_size)
        else:
            data_dict = generate_train_data(data, all_features, label_name, val_size, test_size)
    
        for group in feature_groups:
            numerical_feat = [feat for feat in feature_description_dict[group] if feat not in cat_features]
            cat_feat = [feat for feat in feature_description_dict[group] if feat in cat_features]
            #Train model
            model = train_model(data_dict['train'][0], data_dict['train'][1], numerical_features=numerical_feat, cat_features=cat_feat,  metric_period_logging=500)
            # Make test prediction
            test_pred = make_pred(model, data_dict['test'][0][cat_feat+numerical_feat], proba=False)
            test_pred_proba = make_pred(model, data_dict['test'][0][cat_feat+numerical_feat], proba=True)
            print(f"Evaluating feature group {group}")
            perf = evaluate_performance(test_pred, test_pred_proba, data_dict['test'][1])
            perf_dict[group] = {key: val+[perf[key]] for key, val in perf_dict[group].items()}
            
    perf_dict_average = {group: 
        {metric: np.mean(val) for metric, val in res.items()} 
        for group, res in perf_dict.items()}
    #Plot results
    plot_feature_group_performance(perf_dict_average)
    plot_feature_importance(feature_description_dict, model)
    if perform_t_test:
        test_feature_influence(perf_dict, 'matthew')
    

def test_feature_influence(perf_dict, metric):
    '''
    Perform t-test to check whether there is a significant difference in performance between feature groups
    Use 
    '''
    #Create all pairwise feature group combinations and compute their t-test p-value
    #tttest returns tuple with test_statistic-value and p-value. We are only interested 
    #in p value therefore index first element
    p_values = [ttest_ind(x,y)[1] for x,y in itertools.combinations([res[metric] for res in perf_dict.values()], 2)]
    groups = list( itertools.combinations( perf_dict.keys(), 2))
    #print the results
    for groups, p in zip(groups, p_values):
        print(f' Group {groups[0]} has a different {metric} performance than group {groups[1]} with p-value {p}') 
    
    
    