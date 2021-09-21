import sys

sys.path.append('../loader')
from datasets.unaligned_data_loader import UnalignedDataLoader


def dataset_read_seattle(experiment_setup, batch_size):
    S = {}
    S_val = {}
    S_test = {}
    T = {}
    T_val = {}
    T_test = {}

    # Source and target training data
    S['imgs'] = experiment_setup['source']['train'][0]
    S['labels'] = experiment_setup['source']['train'][1]
    T['imgs'] = experiment_setup['target']['train'][0]
    T['labels'] = experiment_setup['target']['train'][1]

    # source and target val
    S_val['imgs'] = experiment_setup['source']['val'][0]
    S_val['labels'] = experiment_setup['source']['val'][1]
    T_val['imgs'] = experiment_setup['target']['val'][0]
    T_val['labels'] = experiment_setup['target']['val'][1]

    # source and train test
    S_test['imgs'] = experiment_setup['source']['test'][0]
    S_test['labels'] = experiment_setup['source']['test'][1]
    T_test['imgs'] = experiment_setup['target']['test'][0]
    T_test['labels'] = experiment_setup['target']['test'][1]

    # input the data to a torch object, so it is aligned
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, seattle=True)
    dataset = train_loader.load_data()

    val_loader = UnalignedDataLoader()
    val_loader.initialize(S_val, T_val, batch_size, seattle=True)
    dataset_val = val_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, seattle=True)
    dataset_test = test_loader.load_data()

    return dataset, dataset_val, dataset_test
