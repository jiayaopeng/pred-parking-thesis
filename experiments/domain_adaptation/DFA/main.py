from __future__ import print_function
import argparse
import torch
from solver import Solver
import os
import time
import datetime
import json
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')

parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')

parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')

parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')

parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')

parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')

parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')

parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--source', type=str, default='mnist', metavar='N',
                    help='source dataset')

parser.add_argument('--target', type=str, default='usps', metavar='N', help='target dataset')

parser.add_argument('--lambda_1', type=float, default=0.01,
                    help='weight for KL-divergence (default: 0.01)')

parser.add_argument('--lambda_2', type=float, default=10.0,
                    help='weight for DAL (default: 10.0)')

parser.add_argument('--source_only', type=int, default=0,
                    help='create results for training only source and transfering')

parser.add_argument('--n_experiments', type=int, default=50,
                    help='Number of repetition of experiment')

parser.add_argument('--include_pbp', type=int, default=0,
                    help='whether to include information about pbp parking transaction')

parser.add_argument('--area_cv', type=int, default=1,
                   help='cross validation with all areas in seattle ')

parser.add_argument('--experiment_name', type=str, default='',
                    help='name to append to the experiment folder')
# newly added params
parser.add_argument('--use_batchnorm', type=int, default=0,
                    help='use the batch normal layer')

parser.add_argument('--hidden_dim', type=int, default=64,
                    help='choose the number of hidden units in the neural network')

parser.add_argument('--early_stop', type=int, default=1,
                    help='whether to use early stopping')

parser.add_argument('--patience', type=int, default=10,
                    help='whether to use patience')

parser.add_argument('--sm_mode', type=int, default=0,
                    help='whether to use sagemaker mode')

parser.add_argument('--output_dim', type=int, default=32,
                    help='the output dimensions')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def main():
    seattle_areas = ['Greenlake', 'South Lake Union', 'Commercial Core', 'Pike-Pine', 'Uptown', 'Ballard', 'First Hill',
                     'Chinatown/ID', 'Pioneer Square']
    experiment_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + args.experiment_name

    if args.area_cv:
        all_area_combinations = [{'Source': [area for area in seattle_areas if area != test_area],
                               'Target': [test_area]}
                               for test_area in seattle_areas]
    else:
        all_area_combinations = [{'Source': ['Greenlake', 'South Lake Union', 'Commercial Core', 'Pike-Pine', 'Ballard',
                                             'First Hill', 'Chinatown/ID', 'Pioneer Square'],
                                  'Target': ["Uptown"]}]
    
    # Below log the mean of the lists for the validation matthew score - this is is the score we optimize against in hyperparameter tuning
    val_ls = [] 
    
    # Below holds result for test: for each area, save to a dictionary to hold the test result of the metic
    mathew_ls_ls = {}
    auc_ls_ls = {}
    fbeta_ls_ls = {}
    f1_ls_ls = {}
    accu_ls_ls = {}
    pres_ls_ls = {}
    recall_ls_ls = {}
    
    # Below holds the result for val: for each area, save to a dictionary to hold the valid result of the matthew
    val_matthew_ls_ls = {} 

    for area_split in all_area_combinations:
        # save the record file to the record directory, if the time dir to store the record file does not exist, then make dir
        record_num = 0
        record_file = f"record/{experiment_folder}/test_{','.join(area_split['Source']).replace('/', '_')}_{','.join(area_split['Target']).replace('/', '_')}_{record_num}.json"
        record_file_val = f"record/{experiment_folder}/test_{','.join(area_split['Source']).replace('/', '_')}_{','.join(area_split['Target']).replace('/', '_')}_{record_num}_validation.json"

        while os.path.exists(record_file):
            record_num += 1
            record_file = f"record/{experiment_folder}/test_{','.join(area_split['Source']).replace('/', '_')}_{','.join(area_split['Target']).replace('/', '_')}_{record_num}.json"
            record_file_val = f"record/{experiment_folder}/test_{','.join(area_split['Source']).replace('/', '_')}_{','.join(area_split['Target']).replace('/', '_')}_{record_num}_validation.json"

        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        if not os.path.exists(f'record/{experiment_folder}'):
            os.mkdir(f'record/{experiment_folder}')

        # Test output: for that particular area combination, save the list of metric values
        auc_ls = []
        mathew_ls = []
        fbeta_ls = []
        f1_ls = []
        accu_ls = []
        pres_ls = []
        recall_ls = []
        
        # Valid output
        val_matthew_ls = [] # the validation result for matthew

        for i in range(args.n_experiments * (not args.source_only)): # determine how many experiments we are running

            solver = Solver(args, device, source=area_split['Source'], target=area_split['Target'],
                            optimizer=args.optimizer, num_k=args.num_k,
                            checkpoint_dir=args.checkpoint_dir,
                            save_epoch=args.save_epoch)
            counter = 0
            best_matthew = -1  # initialize Matthew, and that is what we optimize against
            for t in range(args.max_epoch):
                t0 = time.time()
                # train with all source data and 75% of target data(without accessing labels)
                num = solver.train(t, record_file=None)
                print('{} seconds'.format(time.time() - t0))
                # valid with the 25 % validation data from the target to select hyperparameters
                val_out = solver.test(t, solver.dataset_val,  test_or_val='val', record_file=record_file_val, save_model=args.save_model)
                # test with the 75% data from target (with labels)
                test_out = solver.test(t, solver.datasets, test_or_val='test', record_file=record_file, save_model=args.save_model)

                # early stopping logic
                # when counter is bigger than the patience, stop and save
                # the best test output at that time
                if val_out[-1] > best_matthew:  # check if current accuracy on validation set is better than best accuracy
                    best_matthew = val_out[-1] # save the current validation AUC as the best
                    # save the best test AUC and other params at the time
                    best_auc_test = test_out['AUC'][-1]
                    best_matthew_test = test_out['Matthew'][-1]
                    best_fbeta_test = test_out['FBeta'][-1]
                    best_f1_test = test_out['F1'][-1]
                    best_accuracy_test = test_out['Accuracy'][-1]
                    best_precision_test = test_out['Precision'][-1]
                    best_recall_test = test_out['Recall'][-1]
                    counter = 0
                else:
                    counter += 1
                if args.early_stop:  # if we have no better accuracy on validation set #args.patience times in a row, apply early stoppinng
                    if counter > args.patience:
                        break
            # after all the epochs(maybe stopped in between), save the last validation matthew to the list
            # For val: and all the other best metrics saved to a list
            val_matthew_ls.append(best_matthew) # for all the epochs of that experiment, get the best matthew out of all epochs
            
            # For test: list of every epoch result (one experiments)
            auc_ls.append(best_auc_test) 
            mathew_ls.append(best_matthew_test)
            fbeta_ls.append(best_fbeta_test)
            f1_ls.append(best_f1_test)
            accu_ls.append(best_accuracy_test)
            pres_ls.append(best_precision_test)
            recall_ls.append(best_recall_test)

        # the below holds the best validation Matthew for all the experiments
        # first val_matthew_ls holds the auc for all the experiments run in this area split setup, get the mean of all the
        # experiment's matthew
        # log the avg matthew, first here we are taking the mean of matthew for each epoch
        val_ls.append(np.mean(val_matthew_ls))  # val_ls the holder for avg matthew of all the experiment
        
        # save the valid result matthew for each area
        val_matthew_ls_ls[str(area_split)] = val_matthew_ls  # save for each area, all the experiments in that area
        
        # save the test result for each area of different metric
        mathew_ls_ls[str(area_split)] = mathew_ls  # this is for test
        auc_ls_ls[str(area_split)] = auc_ls 
        accu_ls_ls[str(area_split)] = accu_ls
        f1_ls_ls[str(area_split)] = f1_ls
        fbeta_ls_ls[str(area_split)] = fbeta_ls
        pres_ls_ls[str(area_split)] = pres_ls
        recall_ls_ls[str(area_split)] = recall_ls

    # log the avg matthew, second here we are taking the mean of the matthew for all the areas
    print(f"Average Matthew: {np.mean(val_ls)};") # best epoch result, mean over experiments, mean over areas

    # dump to json
    if not args.source_only:
        if args.sm_mode:
            # save output to opt/ml/model for final training run
            output_file = f"/opt/ml/model/test_parking.json"  # /opt/ml/model fixed the path of the output, for the final run
        else:
            output_file = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") +  f"domain_adaptation.json"
    else:
        if args.sm_mode:
            # save output to opt/ml/model for final training run
            output_file = f"/opt/ml/model/test_parking_source_only.json"  # /opt/ml/model fixed the path of the output, for the final run
        else:
            output_file = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") +  f"source_only.json"

    result_dict = {
        "auc": auc_ls_ls,
        "matthew": mathew_ls_ls,
        "f1": f1_ls_ls,
        "fbeta": fbeta_ls_ls,
        "accuracy": accu_ls_ls,
        'precision': pres_ls_ls,
        'recall': recall_ls_ls,
        'val_matthew': val_matthew_ls_ls  # val
    }
    
    with open(output_file, 'w') as outfile:
        json.dump(result_dict, outfile)


if __name__ == '__main__':
    main()
