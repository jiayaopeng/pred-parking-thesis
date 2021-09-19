# Predictive Parking Analysis

This repository holds code for 4 different experiments for Master Thesis "Predictive Parking Analysis" submitted to Frankfurt School (MADS) by Jiayao Peng.

# Pre-requisites
- Python3
- Pip

# Experiments
## 1. Baseline
TODO: some description (3 sentences) for baseline. Input, process, output.

### How to run it
In order to run the baseline experiment, please run [this notebook](experiments/baseline/baseline.ipynb).


## 2. Location Similarity
TODO: some description (3 sentences) for location similarity. Input, process, output.

### How to run it
In order to run the location similarity experiment, please run [this notebook](experiments/location_similarity/location_similarity.ipynb).


## 3. Domain Adaptation (DFA)
TODO: some description (3 sentences) for Domain Adaptation (DFA). Input, process, output.

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
TODO: some description (3 sentences) for Domain Adaptation (Correlation Alignment). Input, process, output.

```
# Change the working directory to the experiment folder
cd experiments/domain_adaptation/PyTorch-Deep-CORAL

# Prepare environment to execute the experiment code
sh setup.sh

# Run the experiment
sh run_experiments_coral_parking.sh
```
