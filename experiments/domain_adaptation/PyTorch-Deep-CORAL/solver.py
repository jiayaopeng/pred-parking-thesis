from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from sklearn.metrics import roc_auc_score, matthews_corrcoef, fbeta_score, accuracy_score, f1_score, recall_score, \
    precision_score

from coral import coral


def train(model, optimizer, datasets, tracker, args, epoch=0, batch_size=32):
    model.train()
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

    # Trackers to monitor classification and CORAL loss
    classification_loss_tracker = tracker.track('classification_loss', tracker_class(**tracker_params))
    coral_loss_tracker = tracker.track('CORAL_loss', tracker_class(**tracker_params))

    for batch_idx, data in enumerate(datasets):
        target_data = data['T']
        source_data = data['S']
        source_label = data['S_label']
        target_label = data['T_label']
        if source_data.size()[0] < batch_size or target_data.size()[0] < batch_size:
            # skip the last batch
            break

        source_data, source_label = Variable(source_data.to(device=args.device)), Variable(
            source_label.to(device=args.device))
        target_data = Variable(target_data.to(device=args.device))

        optimizer.zero_grad()

        out_source = model(source_data)
        out_target = model(target_data)

        classification_loss = F.cross_entropy(out_source, source_label)

        # This is where the magic happens
        coral_loss = coral(out_source, out_target)
        composite_loss = classification_loss + args.lambda_coral * coral_loss

        composite_loss.backward()
        optimizer.step()

        classification_loss_tracker.append(classification_loss.item())
        coral_loss_tracker.append(coral_loss.item())


def test(model, device, epoch, dataset, result_dict, test_or_val, record_file=None, save_model=False):
    correct3 = 0  # sum of proba, max
    size = 0
    pred = np.asarray([])
    y_true = np.asarray([])
    for batch_idx, data in enumerate(dataset):
        img = data['T']
        label = data['T_label']
        # img, label = img.cuda(), label.long().cuda()
        img, label = img.to(device), label.long().to(device)
        img, label = Variable(img, volatile=True), Variable(label)
        output_ensemble = model(img)
        # eg. two dimensional (30%, 70%)
        pred_ensemble = output_ensemble.data.max(1)[1]  # combination of the two

        pred = np.concatenate([pred, pred_ensemble.cpu().numpy()])
        y_true = np.concatenate([y_true, label.data.cpu().numpy()])
        k = label.data.size()[0]
        correct3 += pred_ensemble.eq(label.data).cpu().sum()
        size += k

    # TODO: change the tuner to same metric as below, below line is what tuner looks at
    if test_or_val == 'test':
        print(
            f'Test set: Matthew: {matthews_corrcoef(y_true=y_true, y_pred=pred)};')

        result_dict['AUC'] = result_dict['AUC'] + [roc_auc_score(y_score=pred, y_true=y_true)]
        result_dict['Matthew'] = result_dict['Matthew'] + [matthews_corrcoef(y_true=y_true, y_pred=pred)]
        result_dict['F1'] = result_dict['F1'] + [f1_score(y_true=y_true, y_pred=pred)]
        result_dict['FBeta'] = result_dict['FBeta'] + [fbeta_score(y_true=y_true, y_pred=pred, beta=0.33)]
        result_dict['Accuracy'] = result_dict['Accuracy'] + [accuracy_score(y_true=y_true, y_pred=pred)]
        result_dict['Precision'] = result_dict['Precision'] + [precision_score(y_true=y_true, y_pred=pred)]
        result_dict['Recall'] = result_dict['Recall'] + [recall_score(y_true=y_true, y_pred=pred)]

        if record_file:
            with open(record_file, 'w') as outfile:
                json.dump(result_dict, outfile)

        return result_dict

    elif test_or_val == 'val':
        print(
            f'Val set: Matthew: {matthews_corrcoef(y_true=y_true, y_pred=pred)};')
        # this above is what the regex of the tuner is looking at, needed to be matched exactly

        result_dict['AUC'] = result_dict['AUC'] + [roc_auc_score(y_score=pred, y_true=y_true)]
        result_dict['Matthew'] = result_dict['Matthew'] + [matthews_corrcoef(y_true=y_true, y_pred=pred)]
        result_dict['F1'] = result_dict['F1'] + [f1_score(y_true=y_true, y_pred=pred)]
        result_dict['FBeta'] = result_dict['FBeta'] + [fbeta_score(y_true=y_true, y_pred=pred, beta=0.33)]
        result_dict['Accuracy'] = result_dict['Accuracy'] + [accuracy_score(y_true=y_true, y_pred=pred)]
        result_dict['Precision'] = result_dict['Precision'] + [precision_score(y_true=y_true, y_pred=pred)]
        result_dict['Recall'] = result_dict['Recall'] + [recall_score(y_true=y_true, y_pred=pred)]

        if record_file:
            with open(record_file, 'w') as outfile:
                json.dump(result_dict, outfile)

        return result_dict['Matthew']


def set_optimizer(model, args, which_opt='momentum', momentum=0.9):
    if which_opt == 'momentum':
        opt_coral = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=momentum)

    if which_opt == 'adam':
        opt_coral = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    return opt_coral
