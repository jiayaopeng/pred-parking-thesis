import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse


# TODO: doing the hyperparameter on different archtetue, and note when change generate network, also change the shape of z for latent dimensions
# so match z with the last layer of the generator
class Feature(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_batchnorm, prob=0.5):
        super(Feature, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1_fc = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2_fc = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn3_fc = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.prob = prob
        # layers to reconstruct/reverse the features
        self.rev_fc1 = nn.Linear(output_dim, hidden_dim)
        self.rev_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rev_fc3 = nn.Linear(hidden_dim, input_dim)
        self.use_batchnorm = use_batchnorm
        
    def decode(self, z, batch_size, output_dim):
        # reconstruction/ reverse operation for the gausian, z is the shape of the Gaussian
        z = z.view(batch_size, output_dim)

        # z = self.unpool1(z, self.indices2)
        # z = self.relu(self.bn1_de_z(self.de_conv1(z)))
        # z = self.unpool2(z, self.indices1)
        # x = self.relu(self.bn2_de_z(self.de_conv2(z)))

        # a = z
        # x = self.relu(self.bn1_res(self.conv1_res(z)))
        # x = self.bn2_res(self.conv2_res(x))
        # x = a + x

        #x = self.relu(self.bn1_fc(self.rev_fc1(z)))
        #x = self.rev_fc2(x)
        #----------
        # TODO: (optional, Jonas did not use weight tying, the weights are differents when we reconstruct the image/features)
        # weight tying could help to reduce the training effort
        #x = self.relu(F.linear(z, torch.linalg.pinv(self.fc3.weight)))
        #x = self.relu(F.linear(x, torch.linalg.pinv(self.fc2.weight)))
        #x = F.linear(x, torch.linalg.pinv(self.fc1.weight))
        x = self.relu(self.rev_fc1(z))
        x = self.relu(self.rev_fc2(x))
        x= self.rev_fc3(x)
        return x

    def forward(self, x, is_deconv=False):
        x = x.view(x.size(0), x.size(1)).float()
        ##x = F.dropout(x, training=self.training, p=self.prob)
        #x = self.relu(self.fc1(x))
        ##x = F.dropout(x, training=self.training, p=self.prob)
        #x = self.relu(self.fc2(x))
        ##x = F.dropout(x, training=self.training, p=self.prob)
        #x = self.fc3(x)

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

        if is_deconv:            
################################################################################        
###### uncomment these lines for adaptation scenario from USPS to MNIST ########
################################################################################         
# =============================================================================
            # residual blocks
            # a = x
            # x = self.relu(self.bn1_res(self.conv1_res(x)))
            # x = self.bn2_res(self.conv2_res(x))
            # x = a + x
            # b = x
            # x = self.relu(self.bn3_res(self.conv3_res(x)))
            # x = self.bn4_res(self.conv4_res(x))
            # x = b + x     
            # c = x
            # x = self.relu(self.bn5_res(self.conv5_res(x)))
            # x = self.bn6_res(self.conv6_res(x))
            # x = c + x
            # d = x
            # x = self.relu(self.bn7_res(self.conv7_res(x)))
            # x = self.bn8_res(self.conv8_res(x))
            # x = d + x            
# =============================================================================
            #x = self.relu(F.linear(x, torch.linalg.pinv(self.fc3.weight)))
            #x = self.relu(F.linear(x, torch.linalg.pinv(self.fc2.weight)))
            #x = F.linear(x, torch.linalg.pinv(self.fc1.weight))
            x = self.relu(self.rev_fc1(x))
            x = self.relu(self.rev_fc2(x))
            x= self.rev_fc3(x)
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim, output_dim, class_dim, use_batchnorm, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(output_dim, output_dim)
        self.bn1_fc = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, hidden_dim)
        self.bn2_fc = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, class_dim)
        self.bn3_fc = nn.BatchNorm1d(class_dim)
        self.relu = nn.ReLU()
        self.prob = prob
        self.use_batchnorm = use_batchnorm

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = x.view(x.size(0), x.size(1))
        #x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn1_fc(self.fc1(x)) if self.use_batchnorm else self.fc1(x))
        #x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn2_fc(self.fc2(x)) if self.use_batchnorm == 1 else self.fc2(x))
        #x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        #x = self.fc3(self.bn3_fc(self.fc3(x))) # this is newly added
        return x
