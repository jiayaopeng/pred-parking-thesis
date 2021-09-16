from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import Generator, Classifier
from datasets.dataset_read import dataset_read_seattle
from datasets.seattle import load_seattle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, matthews_corrcoef, fbeta_score, accuracy_score, f1_score, recall_score, \
    precision_score
import json
import os


class Solver(object):
    def __init__(self, args, device, source='seattle',
                 target='seattle',  interval=100, optimizer='adam'
                 , num_k=4, checkpoint_dir=None, save_epoch=10,
                 bucket='s3://vwfs-pred-park-irland/', input_dim=21, # input dim match the number of features
                class_dim=2):
        self.device = device
        self.batch_size = args.batch_size

        if args.include_pbp == 1:
            self.input_dim = input_dim + 1
        # network params
        else:
            self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim  # last layer's output no. of neurons
        self.class_dim = class_dim

        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.writer = SummaryWriter()  # tensor board, Loss per epoch, see how loss evolves , how fast convengeging	
        self.include_pbp = args.include_pbp
        # TODO: add precision and recall
        self.result_dict = {metric: [] for metric in ['AUC', 'Matthew', 'F1', 'FBeta', 'Accuracy', 'Precision', 'Recall']}
        # # get the property/attributes of solve object(without the callable function without the getter function),	
        # Purpose: check the values set that leads to the result
        self.result_dict['props'] = {key: value for key, value in self.__dict__.items()
                                     if not callable(value) and not key.startswith('__') and (
                                             type(value) == int or type(value) == str)}
        # validation dictionary
        self.result_dict_val = {metric: [] for metric in ['AUC', 'Matthew', 'F1', 'FBeta', 'Accuracy', 'Precision', 'Recall']}
        # # get the property/attributes of solve object(without the callable function without the getter function),
        # Purpose: check the values set that leads to the result
        self.result_dict_val['props'] = {key: value for key, value in self.__dict__.items()
                                     if not callable(value) and not key.startswith('__') and (
                                             type(value) == int or type(value) == str)}
        ## data prep for images data, not used for parking
        self.scale = False
        # return the dictionary of PU and PN
        self.bucket = bucket
        self.use_batchnorm = args.use_batchnorm
        self.sm_mode = args.sm_mode
        seattle_data = load_seattle(source_areas=self.source, target_areas=self.target,
                                    include_pbp=self.include_pbp, bucket=self.bucket, sm_mode=self.sm_mode)
        # self.dataset is the train sets from both source and target
        # self.dataset_val in the target, test not there
        self.datasets, self.dataset_val, self.dataset_test = dataset_read_seattle(seattle_data,
                                                                self.batch_size)  # objects for loading data
        
        # (args.batch_size, number of features that G outputs which are latent features (no.of data sample, no. of featurs in latent space)
        # Below is the shape for the gausian prior
        # args.batch_size us the number of the datapoints, and self.output_dim the dimensions of the gausian, the gausian prior is constructed in a self.output_dim dimensional space
        self.z_shape = (self.batch_size, self.output_dim)  # hardcoded number of features here
        print('load finished!')

        self.G = Generator(source=source, target=target, input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                           output_dim=self.output_dim, use_batchnorm=self.use_batchnorm)
        self.C1 = Classifier(source=source, target=target, hidden_dim=self.hidden_dim,output_dim=self.output_dim,
                             class_dim=self.class_dim, use_batchnorm=self.use_batchnorm)  # two classifier are using same network
        self.C2 = Classifier(source=source, target=target, hidden_dim=self.hidden_dim,output_dim=self.output_dim,
                             class_dim=self.class_dim, use_batchnorm=self.use_batchnorm)

        self.G.to(self.device)
        self.C1.to(self.device)
        self.C2.to(self.device)

        # interval here used later for printing
        self.interval = interval
        self.lr = args.lr
        self.set_optimizer(which_opt=optimizer)

        
    def set_optimizer(self, which_opt='momentum', momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=self.lr, weight_decay=0.0005,
                                   momentum=momentum)
            
            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=self.lr, weight_decay=0.0005,
                                    momentum=momentum)
            
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=self.lr, weight_decay=0.0005,
                                    momentum=momentum)
            
        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=self.lr, weight_decay=0.0005)
            
            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=self.lr, weight_decay=0.0005)
            
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=self.lr, weight_decay=0.0005)
            
    # pytorch manually reset gradietn after train
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        
    def ent(self, output):
        # below entropy, 1e-6 make sure not log 0
        return - torch.mean(output * torch.log(output + 1e-6))
    
    def discrepancy(self, out1, out2):
        # descrepancy between the two classifier
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
    
    def train(self, epoch, record_file=None):
        '''
        Train and optimize based on loss
        '''
        print(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        # initialze a L1 loss for distribution alignment
        criterionConsistency = nn.L1Loss().to(self.device)
        # TODO: here check if the generator needs to be trained in paper
        self.G.train()
        self.C1.train()
        self.C2.train()
        if str(self.device) =='cuda':
            torch.cuda.manual_seed(1)
            Tensor = torch.cuda.FloatTensor
        else:
            torch.manual_seed(1)
            Tensor = torch.FloatTensor

        for batch_idx, data in enumerate(self.datasets):
            """
            For the self.datasets, where we had the train from source and target, so we use both of them as train.
            The train from the target does not have access to the labels as it will be used as test dataset later
            """
            # access the target and source train
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            label_t = data['T_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                # skip the last batch 
                break

            img_s = img_s.to(self.device)
            img_t = img_t.to(self.device)
            label_s = Variable(label_s.long().to(self.device))
            label_t = Variable(label_t.long().to(self.device))

            ## The below is the Gausian Variable generated based on the dimensions
            z = Variable(Tensor(np.random.normal(0, 1, self.z_shape)))
            
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            
            ## Use NN to generate latent features from the generator
            feat_s = self.G(img_s)
            feat_t = self.G(img_t)
            ## Use NN to classfify the latent feature
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            ## z_shape[1] is number of features, below reshapes the features, -1 number of features will be inferred
            ## align the latent feature distribution with the gausian
            feat_s_kl = feat_s.view(-1, self.z_shape[1])
            
            # calculate the distribution alignment loss between the gausian and the latent feature distribution with kl
            # here we could also add GPU later if neded
            loss_kld = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(z))
            # calculate the loss of two classifiers on the latent features
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            # final weighted loss (add the two classifier's loss plus the gausian alignment loss
            
            loss_s = loss_s1 + loss_s2 + self.lambda_1 * loss_kld
            
            ## based on the loss computed, optimize the steps
            loss_s.backward()
            self.opt_g.step()  # updates param newwork
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            self.writer.add_scalar("Loss/s1", loss_s1, epoch)
            self.writer.add_scalar("Loss/s2", loss_s2, epoch)
            
            ## original calcualte for updating reconstruction, because previous one has been trained, and updated
            self.writer.add_scalar("Loss/kld", loss_kld, epoch)
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)
            
            feat_s_kl = feat_s.view(-1, self.z_shape[1])
            loss_kld = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(z))
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2 + self.lambda_1 * loss_kld
            loss_dis = self.discrepancy(output_t1, output_t2)
            self.writer.add_scalar("Loss/dis", loss_dis, epoch)
            loss = loss_s - loss_dis
            self.writer.add_scalar("Loss/loss-loss_dis", loss, epoch)
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            
            ## contructing reconstruction loss
            for i in range(self.num_k):
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                
                # get x_rt
                # decoder/ deconvolution , try to get original feature back, here note the is_deconv = True
                # after generator, we reconstruct the features back
                feat_t_recon = self.G(img_t, is_deconv=True)
                # TODO: deconvolve, weight tying, Jonas has removed the weight tying part, but could try to use inverse matrix or something for weight tying
                # reconstruct the gausian, to make sure ggausia space has same dributions as the img above
                # after generator, we reconstruct based on the gausian
                feat_z_recon = self.G.decode(z, self.batch_size, self.output_dim)
                
                # distribution alignment loss (alignment of the reconstructed original images and the reconstruction based on gausian
                loss_dal = criterionConsistency(feat_t_recon, feat_z_recon)	
                self.writer.add_scalar("Loss/distr_consistency_loss", loss_dal, epoch)
                # updated loss function  (also considers the two classifier , if random, two outputs of two classifier will be high	
                loss_dis = self.discrepancy(output_t1, output_t2) + self.lambda_2 * loss_dal
                # optimize
                self.writer.add_scalar("Loss/distr_alignment_loss", loss_dis, epoch)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx
            
            if batch_idx % self.interval == 0:  # print at n. of epochs
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.item(), loss_s2.item(), loss_dis.item()))
                if record_file:	
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s1.item(), loss_s2.item()))
                    record.close()
            #torch.save(self.G,
            #           '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        return batch_idx
    
    def test(self, epoch, dataset, test_or_val, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0  # classifer 1 performance
        correct2 = 0
        correct3 = 0  # sum of proba, max
        size = 0
        pred = np.asarray([])
        y_true = np.asarray([])
        for batch_idx, data in enumerate(dataset):
            """
                In the case of validation data, we pass solver.dataset_val as params, and we will only choose the target data
            from the dataset_val
            
                In the case of test data, we pass solver.datasets as params, and we only use the target data as test(Even
            we have used the target data as train but we did not use the label of it, therefore, it could be used as test 
            with the access to the labels)
            """
            img = data['T']	
            label = data['T_label']	
            #img, label = img.cuda(), label.long().cuda()
            img, label = img.to(self.device), label.long().to(self.device)
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            # eg. two dimensional (30%, 70%)
            test_loss += F.nll_loss(output1, label).item()
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]  # take the max of the class
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]  # combination of the two

            pred = np.concatenate([pred, pred_ensemble.cpu().numpy()])
            y_true = np.concatenate([y_true, label.data.cpu().numpy()])
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        test_loss = test_loss / size  # all loss over no. of mini batch
        # save the result
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(	
                test_loss, correct1, size,	
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        print(
            f'\nTest set: AUC:{roc_auc_score(y_score=pred, y_true=y_true)}, Matthew: {matthews_corrcoef(y_true=y_true, y_pred=pred)} \n')
        # TODO: change the tuner to same metric as below, below line is what tuner looks at
        if test_or_val == 'test':
            print(
                f'Test set: Matthew: {matthews_corrcoef(y_true=y_true, y_pred=pred)};')

            self.result_dict['AUC'] = self.result_dict['AUC'] + [roc_auc_score(y_score=pred, y_true=y_true)]
            self.result_dict['Matthew'] = self.result_dict['Matthew'] + [matthews_corrcoef(y_true=y_true, y_pred=pred)]
            self.result_dict['F1'] = self.result_dict['F1'] + [f1_score(y_true=y_true, y_pred=pred)]
            self.result_dict['FBeta'] = self.result_dict['FBeta'] + [fbeta_score(y_true=y_true, y_pred=pred, beta=0.33)]
            self.result_dict['Accuracy'] = self.result_dict['Accuracy'] + [accuracy_score(y_true=y_true, y_pred=pred)]
            self.result_dict['Precision'] = self.result_dict['Precision'] + [precision_score(y_true=y_true, y_pred=pred)]
            self.result_dict['Recall'] = self.result_dict['Recall'] + [recall_score(y_true=y_true, y_pred=pred)]
            if record_file:
                with open(record_file, 'w') as outfile:
                    json.dump(self.result_dict, outfile)
            return self.result_dict

        elif test_or_val == 'val':
            print(
                f'Val set: Matthew: {matthews_corrcoef(y_true=y_true, y_pred=pred)};')
            # this above is what the regex of the tuner is looking at, needed to be matched exactly

            self.result_dict_val['AUC'] = self.result_dict_val['AUC'] + [roc_auc_score(y_score=pred, y_true=y_true)]
            self.result_dict_val['Matthew'] = self.result_dict_val['Matthew'] + [matthews_corrcoef(y_true=y_true, y_pred=pred)]
            self.result_dict_val['F1'] = self.result_dict_val['F1'] + [f1_score(y_true=y_true, y_pred=pred)]
            self.result_dict_val['FBeta'] = self.result_dict_val['FBeta'] + [fbeta_score(y_true=y_true, y_pred=pred, beta=0.33)]
            self.result_dict_val['Accuracy'] = self.result_dict_val['Accuracy'] + [accuracy_score(y_true=y_true, y_pred=pred)]
            self.result_dict_val['Precision'] = self.result_dict_val['Precision'] + [precision_score(y_true=y_true, y_pred=pred)]
            self.result_dict_val['Recall'] = self.result_dict_val['Recall'] + [recall_score(y_true=y_true, y_pred=pred)]
            if record_file:
                with open(record_file, 'w') as outfile:
                    json.dump(self.result_dict_val, outfile)

            return self.result_dict_val['Matthew']

    def train_source_only(self, epoch, record_file=None):
        '''
            Train only on source domain and transfer the final model to target 	
            '''
        criterion = nn.CrossEntropyLoss().to(self.device)

        # TODO: here check if the generator needs to be trained in paper
        self.G.train()
        self.C1.train()
        self.C2.train()

        if str(self.device) =='cuda':
            torch.cuda.manual_seed(1)
            Tensor = torch.cuda.FloatTensor
        else:
            torch.manual_seed(1)
            Tensor = torch.FloatTensor
        
        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            label_t = data['T_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            img_s = img_s.to(self.device)
            img_t = img_t.to(self.device)
            label_s = Variable(label_s.long().to(self.device))
            label_t = Variable(label_t.long().to(self.device))
            
            z = Variable(Tensor(np.random.normal(0, 1, self.z_shape)))

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            feat_t = self.G(img_t)
      
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_s_kl = feat_s.view(-1, self.z_shape[1])
            # if needed the loss could add the GPU
            loss_kld = F.kl_div(F.log_softmax(feat_s_kl), F.softmax(z))
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)

            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            
        return batch_idx