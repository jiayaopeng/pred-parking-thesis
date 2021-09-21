import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse


# TODO: doing the hyperparameter on different archtetue, and note when change generate network, also change the shape of z for latent dimensions
# so match z with the last layer of the generator
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_batchnorm, prob=0.5):
        super(Model, self).__init__() # inheritance
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1_fc = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2_fc = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn3_fc = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.prob = prob
        # layers to reconstruct/reverse the features

        self.use_batchnorm = use_batchnorm
        self.fc_final = nn.Linear(output_dim, 2)


    def forward(self, x, is_deconv=False):
        x = x.view(x.size(0), x.size(1)).float()
        # linear
        x = self.fc1(x)
        # batch and relu
        x = self.relu(self.bn1_fc(x) if self.use_batchnorm else x)

        # linear
        x = self.fc2(x)
        # batch and relu
        x = self.relu(self.bn2_fc(x) if self.use_batchnorm else x)

        # linear
        x = self.fc3(x)
        # batch and relu
        x = self.relu(self.bn3_fc(x) if self.use_batchnorm else x)

        # final layer
        x = self.fc_final(x)

        return x