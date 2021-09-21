# Predictive Parking Analysis

This repository holds code for 4 different experiments for Master Thesis "Predictive Parking Analysis" submitted to Frankfurt School (MADS) by Jiayao Peng. Below is the structure of the scripts and repository.



# Pre-requisites
- Python3
- Pip

# Experiments
## 1. Baseline
Baseline folder stores the util module where all the helper functions have been stored, and the baseline.ipynb notebook analyzes the result of baseline models. We have build three baseline models including logistic regression, random forest and catboost. The baseline models have been tested in below data settings: all seattle data, only nine study areas and only four study areas. We have also experimented with model performance based on different radius size. 

### How to run it
In order to run the baseline experiment, please run [this notebook](experiments/baseline/baseline.ipynb).


## 2. Location Similarity
For location similarity, we firstly define the location simialrity based on street similarity and cluster representation similarity. Then, construct the cluster representations. Furthermore, create clusters separately in train and test areas. Finally, train a model on matched cluster pairs(use model trained on the train cluster to the matched test cluster). The input is the preprocess data.

### How to run it
In order to run the location similarity experiment, please run [this notebook](experiments/location_similarity/location_similarity.ipynb).


## 3. Domain Adaptation (DFA)
This part we adapted the code from . The idea is to align the source and target data under the alignment of a constructed Gaussian prior. The main script is the main.py.

### How to run it
```
# Change the working directory to the folder of the experiment
cd experiments/domain_adaptation/DFA

# Change the working directory to the experiment folder
sh setup.sh

# Run the experiment
sh run_experiments_parking.sh
```

## 4. Domain Adaptation (Correlation Alignment)
The idea of Deep CORAL is to align the second-order statistics of the source and target domain. This is achieved by adding a CORAL loss of the activation function to the archtecture. The main script to run is train.py

```
# Change the working directory to the experiment folder
cd experiments/domain_adaptation/PyTorch-Deep-CORAL

# Prepare environment to execute the experiment code
sh setup.sh

# Run the experiment
sh run_experiments_coral_parking.sh
```
