import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from datasets.datasets import Dataset

## one loop , loop through source and target
class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}

# TODO: ?? parking does not need below? why we need to normalize, what did Jonas do, why need the logic of the else condition
class UnalignedDataLoader():
    def initialize(self, source, target, batch_size, scale=32, seattle=False):

        dataset_source = Dataset(source['imgs'], source['labels'], seattle=seattle)
        dataset_target = Dataset(target['imgs'], target['labels'], seattle=seattle)

        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size,
            shuffle=True, # shuffle the data
            num_workers=0)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        # paired data so that we can loop through train and test together
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
