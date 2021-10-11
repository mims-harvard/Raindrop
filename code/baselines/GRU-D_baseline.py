"""
Code originates from GRUD_mean.ipynb from GitHub repository https://github.com/Han-JD/GRU-D.
"""

from models import GRUD
import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from utils_phy12 import random_sample


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_dataloader(dataset, outcomes, upsampling_batch, batch_size, split_type, feature_removal_level, missing_ratio,
                    train_proportion=0.8, dev_proportion=0.1, dataset_name='P12'):
    # 80% train, 10% validation, 10% test

    if split_type == 'random':
        # np.random.seed(77)   # if you want the same permutation for each run
        # shuffle data
        permuted_idx = np.random.permutation(dataset.shape[0])
        dataset = dataset[permuted_idx]
        outcomes = outcomes[permuted_idx]

        train_index = int(np.floor(dataset.shape[0] * train_proportion))
        dev_index = int(np.floor(dataset.shape[0] * (train_proportion + dev_proportion)))

        # split dataset to train/dev/test set
        if upsampling_batch:
            train_data = []
            train_label = []
            idx_0 = np.where(outcomes[:train_index, :] == 0)[0]
            idx_1 = np.where(outcomes[:train_index, :] == 1)[0]
            for i in range(train_index // batch_size):
                indices = random_sample(idx_0, idx_1, batch_size)
                train_data.extend(dataset[indices, :, :, :])
                train_label.extend(outcomes[indices, :])
            train_data = np.array(train_data)
            train_label = np.array(train_label)
        else:
            train_data, train_label = dataset[:train_index, :, :, :], outcomes[:train_index, :]

        dev_data, dev_label = dataset[train_index:dev_index, :, :, :], outcomes[train_index:dev_index, :]
        test_data, test_label = dataset[dev_index:, :, :, :], outcomes[dev_index:, :]
    elif split_type == 'age' or split_type == 'gender':
        # # calculate and save statistics
        # idx_under_65 = []
        # idx_over_65 = []
        # idx_male = []
        # idx_female = []
        #
        # P_list = np.load('../../P12data/processed_data/P_list.npy', allow_pickle=True)
        #
        # for i in range(len(P_list)):
        #     age, gender, _, _, _ = P_list[i]['static']
        #     if age > 0:
        #         if age < 65:
        #             idx_under_65.append(i)
        #         else:
        #             idx_over_65.append(i)
        #     if gender == 0:
        #         idx_female.append(i)
        #     if gender == 1:
        #         idx_male.append(i)
        #
        # np.save('saved/grud_idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
        # np.save('saved/grud_idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
        # np.save('saved/grud_idx_male.npy', np.array(idx_male), allow_pickle=True)
        # np.save('saved/grud_idx_female.npy', np.array(idx_female), allow_pickle=True)

        if dataset_name == 'P12':
            prefix = 'grud'
        elif dataset_name == 'P19':
            prefix = 'P19'
        elif dataset_name == 'eICU':    # not possible with split_type == 'age'
            prefix = 'eICU'

        if split_type == 'age':
            idx_train = np.load('saved/' + prefix + '_idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load('saved/' + prefix + '_idx_over_65.npy', allow_pickle=True)
        else:   # split_type == 'gender':
            idx_train = np.load('saved/' + prefix + '_idx_male.npy', allow_pickle=True)
            idx_vt = np.load('saved/' + prefix + '_idx_female.npy', allow_pickle=True)

        if upsampling_batch:
            train_data = []
            train_label = []
            idx_0 = idx_train[np.where(outcomes[idx_train, :] == 0)[0]]
            idx_1 = idx_train[np.where(outcomes[idx_train, :] == 1)[0]]
            for i in range(len(idx_train) // batch_size):   # last small batch is dropped
                indices = random_sample(idx_0, idx_1, batch_size)
                train_data.extend(dataset[indices, :, :, :])
                train_label.extend(outcomes[indices, :])
            train_data = np.array(train_data)
            train_label = np.array(train_label)
        else:
            train_data, train_label = dataset[idx_train, :, :, :], outcomes[idx_train, :]

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

        dev_data, dev_label = dataset[idx_val, :, :, :], outcomes[idx_val, :]
        test_data, test_label = dataset[idx_test, :, :, :], outcomes[idx_test, :]

    if feature_removal_level == 'sample':
        num_all_features = dev_data.shape[2]
        num_missing_features = round(missing_ratio * num_all_features)
        for i, patient in enumerate(dev_data):
            idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            patient[:, idx, :] = np.zeros(shape=(dev_data.shape[1], num_missing_features, dev_data.shape[3]))
            dev_data[i] = patient
        for i, patient in enumerate(test_data):
            idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            patient[:, idx, :] = np.zeros(shape=(test_data.shape[1], num_missing_features, test_data.shape[3]))
            test_data[i] = patient
    elif feature_removal_level == 'set':
        num_all_features = dev_data.shape[2]
        num_missing_features = round(missing_ratio * num_all_features)

        if dataset_name == 'P12':
            density_scores = np.load('saved/IG_density_scores_P12.npy', allow_pickle=True)

            inputdict = {
                "ALP": 0,  # o
                "ALT": 1,  # o
                "AST": 2,  # o
                "Albumin": 3,  # o
                "BUN": 4,  # o
                "Bilirubin": 5,  # o
                "Cholesterol": 6,  # o
                "Creatinine": 7,  # o
                "DiasABP": 8,  # o
                "FiO2": 9,  # o
                "GCS": 10,  # o
                "Glucose": 11,  # o
                "HCO3": 12,  # o
                "HCT": 13,  # o
                "HR": 14,  # o
                "K": 15,  # o
                "Lactate": 16,  # o
                "MAP": 17,  # o
                "Mg": 18,  # o
                "Na": 19,  # o
                "PaCO2": 20,  # o
                "PaO2": 21,  # o
                "Platelets": 22,  # o
                "RespRate": 23,  # o
                "SaO2": 24,  # o
                "SysABP": 25,  # o
                "Temp": 26,  # o
                "Tropl": 27,  # o
                "TroponinI": 27,  # temp: regarded same as Tropl
                "TropT": 28,  # o
                "TroponinT": 28,  # temp: regarded same as TropT
                "Urine": 29,  # o
                "WBC": 30,  # o
                "Weight": 31,  # o
                "pH": 32,  # o
                "NIDiasABP": 33,  # unused variable
                "NIMAP": 34,  # unused variable
                "NISysABP": 35,  # unused variable
                "MechVent": 36,  # unused variable
                "RecordID": 37,  # unused variable
                "Age": 38,  # unused variable
                "Gender": 39,  # unused variable
                "ICUType": 40,  # unused variable
                "Height": 41  # unused variable
            }
            idx = []
            for _, name in density_scores:
                if inputdict[name] < 33:
                    idx.append(inputdict[name])
            idx = list(set(idx[:num_missing_features]))
        elif dataset_name == 'P19':
            density_scores = np.load('saved/IG_density_scores_P19.npy', allow_pickle=True)
            idx = list(map(int, density_scores[:, 0][:num_missing_features]))
        elif dataset_name == 'eICU':
            density_scores = np.load('saved/IG_density_scores_eICU.npy', allow_pickle=True)
            idx = list(map(int, density_scores[:, 0][:num_missing_features]))
        elif dataset_name == 'PAM':
            density_scores = np.load('saved/IG_density_scores_PAM.npy', allow_pickle=True)
            idx = list(map(int, density_scores[:, 0][:num_missing_features]))

        dev_data[:, :, idx, :] = np.zeros(shape=(dev_data.shape[0], dev_data.shape[1], len(idx), dev_data.shape[3]))
        test_data[:, :, idx, :] = np.zeros(shape=(test_data.shape[0], test_data.shape[1], len(idx), test_data.shape[3]))

    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset)
    dev_dataloader = utils.DataLoader(dev_dataset)
    test_dataloader = utils.DataLoader(test_dataset)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader


def train_gru_d(num_runs, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, n_epochs,
                batch_size, upsampling_batch, split_type, feature_removal_level, missing_ratio, dataset, predictive_label='mortality'):
    model_path = 'saved/grud_model_best.pt'

    acc_all = []
    auc_all = []
    aupr_all = []
    precision_all = []
    recall_all = []
    F1_all = []

    for r in range(num_runs):
        if dataset == 'P12':
            t_dataset = np.load('saved/dataset.npy')
            if predictive_label == 'mortality':
                t_out = np.load('saved/y1_out.npy')
            elif predictive_label == 'LoS':  # for P12 only
                t_out = np.load('saved/LoS_y1_out.npy')
        elif dataset == 'P19':
            t_dataset = np.load('saved/P19_dataset.npy')
            t_out = np.load('saved/P19_y1_out.npy')
        elif dataset == 'eICU':
            t_dataset = np.load('saved/eICU_dataset.npy')
            t_out = np.load('saved/eICU_y1_out.npy')
        elif dataset == 'PAM':
            t_dataset = np.load('saved/PAM_dataset.npy')
            t_out = np.load('saved/PAM_y1_out.npy')

        if r == 0:
            print(t_dataset.shape, t_out.shape)

        train_dataloader, dev_dataloader, test_dataloader = data_dataloader(t_dataset, t_out, upsampling_batch, batch_size,
                                                                            split_type, feature_removal_level, missing_ratio,
                                                                            train_proportion=0.8, dev_proportion=0.1,
                                                                            dataset_name=dataset)
        if dataset == 'P12':
            x_mean = torch.Tensor(np.load('saved/x_mean_aft_nor.npy'))
            n_classes = 2
        elif dataset == 'P19':
            x_mean = torch.Tensor(np.load('saved/P19_x_mean_aft_nor.npy'))
            n_classes = 2
        elif dataset == 'eICU':
            x_mean = torch.Tensor(np.load('saved/eICU_x_mean_aft_nor.npy'))
            n_classes = 2
        elif dataset == 'PAM':
            x_mean = torch.Tensor(np.load('saved/PAM_x_mean_aft_nor.npy'))
            n_classes = 8
        print(x_mean.shape)

        model = GRUD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                     dropout_type='mloss', x_mean=x_mean, num_layers=num_layers)

        epoch_losses = []

        old_state_dict = {}
        for key in model.state_dict():
            old_state_dict[key] = model.state_dict()[key].clone()

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            criterion = torch.nn.BCELoss()
        elif dataset == 'PAM':
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=1, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

        print('\n------------------\nRUN %d: Training started\n------------------' % r)
        best_aupr_val = 0
        for epoch in range(n_epochs):
            # train the model
            losses, acc = [], []
            label, pred = [], []
            y_pred_col = []
            model.train()
            for train_data, train_label in train_dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                train_data = torch.squeeze(train_data)
                train_label = torch.squeeze(train_label, dim=0)

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    y_pred = model(train_data)

                    # Save predict and label
                    y_pred_col.append(y_pred.item())
                    pred.append(y_pred.item() > 0.5)
                    label.append(train_label.item())

                    # Compute loss
                    loss = criterion(y_pred, train_label)
                    acc.append(
                        torch.eq(
                            (y_pred.data > 0.5).float(),
                            train_label)
                    )
                    losses.append(loss.item())
                elif dataset == 'PAM':
                    y_pred = model(train_data, dataset_name=dataset)

                    # Save predict and label
                    y_pred_col.append(torch.argmax(y_pred).item())
                    label.append(train_label.item())

                    # Compute loss
                    loss = criterion(torch.unsqueeze(y_pred, 0), train_label.type(torch.LongTensor))

                    acc.append(
                        torch.eq(
                            torch.argmax(y_pred),
                            train_label)
                    )
                    losses.append(loss.item())

                loss.backward()
                optimizer.step()

            train_acc = torch.mean(torch.cat(acc).float())
            train_loss = np.mean(losses)

            train_pred_out = pred
            train_label_out = label

            # save new params
            new_state_dict = {}
            for key in model.state_dict():
                new_state_dict[key] = model.state_dict()[key].clone()

            # compare params
            for key in old_state_dict:
                if (old_state_dict[key] == new_state_dict[key]).all():
                    print('Not updated in {}'.format(key))

            # validation loss
            losses, acc = [], []
            label, pred = [], []
            model.eval()
            for dev_data, dev_label in dev_dataloader:
                dev_data = torch.squeeze(dev_data)
                dev_label = torch.squeeze(dev_label, dim=0)

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    y_pred = model(dev_data)

                    # Save predict and label
                    pred.append(y_pred.item())
                    label.append(dev_label.item())

                    # Compute loss
                    loss = criterion(y_pred, dev_label)
                    acc.append(
                        torch.eq(
                            (y_pred.data > 0.5).float(),
                            dev_label)
                    )

                    losses.append(loss.item())
                elif dataset == 'PAM':
                    y_pred = model(dev_data, dataset_name=dataset)

                    # Save predict and label
                    pred.append(torch.argmax(y_pred).item())
                    label.append(dev_label.item())

                    # Compute loss
                    loss = criterion(torch.unsqueeze(y_pred, 0), dev_label.type(torch.LongTensor))

                    acc.append(
                        torch.eq(
                            torch.argmax(y_pred),
                            dev_label)
                    )
                    losses.append(loss.item())

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                auc_val = roc_auc_score(label, pred)
                aupr_val = average_precision_score(label, pred)
            elif dataset == 'PAM':
                label_oh = np.array(label)
                label_oh = one_hot(label_oh[:, np.newaxis])
                pred_oh = np.array(pred)
                pred_oh = one_hot(pred_oh[:, np.newaxis])
                auc_val = roc_auc_score(label_oh, pred_oh)
                aupr_val = average_precision_score(label_oh, pred_oh)

            scheduler.step(aupr_val)  # reduce learning rate when this metric has stopped improving

            if aupr_val > best_aupr_val:
                best_aupr_val = aupr_val
                torch.save(model, model_path)

            dev_acc = torch.mean(torch.cat(acc).float())
            dev_loss = np.mean(losses)

            dev_pred_out = pred
            dev_label_out = label

            print("VALIDATION: Epoch %d, val_acc: %.2f, val_loss: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                  (epoch, dev_acc * 100, dev_loss.item(), aupr_val * 100, auc_val * 100))

            # if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            #     print(confusion_matrix(label, (np.array(pred) > 0.5).astype(int), labels=list(range(n_classes))))
            # elif dataset == 'PAM':
            #     print(confusion_matrix(label, pred, labels=list(range(n_classes))))

            # test loss
            losses, acc = [], []
            label, pred = [], []
            model.eval()
            for test_data, test_label in test_dataloader:
                test_data = torch.squeeze(test_data)
                test_label = torch.squeeze(test_label, dim=0)

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    y_pred = model(test_data)

                    # Save predict and label
                    pred.append(y_pred.item())
                    label.append(test_label.item())

                    # Compute loss
                    loss = criterion(y_pred, test_label)
                    acc.append(
                        torch.eq(
                            (y_pred.data > 0.5).float(),
                            test_label)
                    )
                    losses.append(loss.item())
                elif dataset == 'PAM':
                    y_pred = model(test_data, dataset_name=dataset)

                    # Save predict and label
                    pred.append(torch.argmax(y_pred).item())
                    label.append(test_label.item())

                    # Compute loss
                    loss = criterion(torch.unsqueeze(y_pred, 0), test_label.type(torch.LongTensor))

                    acc.append(
                        torch.eq(
                            torch.argmax(y_pred),
                            test_label)
                    )
                    losses.append(loss.item())

            test_acc = torch.mean(torch.cat(acc).float())
            test_loss = np.mean(losses)

            test_pred_out = pred
            test_label_out = label

            epoch_losses.append([
                 train_loss, dev_loss, test_loss,
                 train_acc, dev_acc, test_acc,
                 train_pred_out, dev_pred_out, test_pred_out,
                 train_label_out, dev_label_out, test_label_out,
             ])

        print('\n------------------\nRUN %d: Training finished\n------------------' % r)

        # Test set
        model = torch.load(model_path)

        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label, dim=0)

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                y_pred = model(test_data)

                # Save predict and label
                pred.append(y_pred.item())
                label.append(test_label.item())

                # Compute loss
                loss = criterion(y_pred, test_label)
                acc.append(
                    torch.eq(
                        (y_pred.data > 0.5).float(),
                        test_label)
                )
                losses.append(loss.item())
            elif dataset == 'PAM':
                y_pred = model(test_data, dataset_name=dataset)

                # Save predict and label
                pred.append(torch.argmax(y_pred).item())
                label.append(test_label.item())

                # Compute loss
                loss = criterion(torch.unsqueeze(y_pred, 0), test_label.type(torch.LongTensor))

                acc.append(
                    torch.eq(
                        torch.argmax(y_pred),
                        test_label)
                )
                losses.append(loss.item())

        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)

        pred = np.asarray(pred)
        label = np.asarray(label)

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            auc_score = roc_auc_score(label, pred)
            aupr_score = average_precision_score(label, pred)
        elif dataset == 'PAM':
            label_oh = np.array(label)
            label_oh = one_hot(label_oh[:, np.newaxis])
            pred_oh = np.array(pred)
            pred_oh = one_hot(pred_oh[:, np.newaxis])
            auc_score = roc_auc_score(label_oh, pred_oh)
            aupr_score = average_precision_score(label_oh, pred_oh)
            precision = precision_score(label, pred, average='macro', )
            recall = recall_score(label, pred, average='macro', )
            F1_score = f1_score(label, pred, average='macro', )

            print("\nTEST: test_precision: %.2f test_recall: %.2f, test_F1: %.2f\n" %
                  (precision * 100, recall * 100, F1_score * 100))

        print("\nTEST: test_acc: %.2f aupr_test: %.2f, auc_test: %.2f\n" %
              (test_acc * 100, aupr_score * 100, auc_score * 100))

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            print(confusion_matrix(label, (np.array(pred) > 0.5).astype(int), labels=list(range(n_classes))))
        elif dataset == 'PAM':
            print(confusion_matrix(label, pred, labels=list(range(n_classes))))

        acc_all.append(test_acc * 100)
        auc_all.append(auc_score * 100)
        aupr_all.append(aupr_score * 100)
        if dataset == 'PAM':
            precision_all.append(precision * 100)
            recall_all.append(recall * 100)
            F1_all.append(F1_score * 100)

    # print mean and std of all metrics
    acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
    mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
    mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
    mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
    print('------------------------------------------')
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))
    if dataset == 'PAM':
        precision_all, recall_all, F1_all = np.array(precision_all), np.array(recall_all), np.array(F1_all)
        mean_precision, std_precision = np.mean(precision_all), np.std(precision_all)
        mean_recall, std_recall = np.mean(recall_all), np.std(recall_all)
        mean_F1, std_F1 = np.mean(F1_all), np.std(F1_all)
        print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
        print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
        print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))

    # #show AUROC on test data for last trained epoch
    # test_preds, test_labels = epoch_losses[-1][8], epoch_losses[-1][11]
    # plot_roc_and_auc_score(test_preds, test_labels, 'GRU-D')


def plot_roc_and_auc_score(outputs, labels, title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve, AREA = {:.4f}'.format(auc_score))
    plt.plot([0,1], [0,1], 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'eICU', 'PAM'])
    parser.add_argument('--withmissingratio', default=False,
                        help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')  #
    parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'],
                        help='only use for P12 and P19')
    parser.add_argument('--reverse', default=False,
                        help='if True,use female, older for tarining; if False, use female or younger for training')  #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal',
                        choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal')  #
    parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                        help='use this only with P12 dataset (mortality or length of stay)')
    args = parser.parse_args(args=[])
    print('Dataset used: ', args.dataset)   # possible values: 'P12', 'P19', 'eICU', 'PAM'

    if args.dataset == 'P12':
        input_size = 33  # num of variables base on the paper
        hidden_size = 33  # same as inputsize
    elif args.dataset == 'P19':
        input_size = 40
        hidden_size = 40
    elif args.dataset == 'eICU':
        input_size = 16
        hidden_size = 16
    elif args.dataset == 'PAM':
        input_size = 17
        hidden_size = 17

    if args.withmissingratio == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # if True, with missing ratio
    else:
        missing_ratios = [0]

    for missing_ratio in missing_ratios:
        num_runs = 1
        if args.dataset == 'P12' or args.dataset == 'P19' or args.dataset == 'eICU':
            output_size = 1
        elif args.dataset == 'PAM':
            output_size = 8
        num_layers = 49  # num of step or layers base on the paper / number of hidden states
        dropout = 0.0  # dropout_type : Moon, Gal, mloss
        learning_rate = 0.001
        n_epochs = 20
        batch_size = 128
        if args.dataset == 'P12' or args.dataset == 'P19' or args.dataset == 'eICU':
            upsampling_batch = True
        elif args.dataset == 'PAM':
            upsampling_batch = False

        split_type = args.splittype  # possible values: 'random', 'age', 'gender'
        reverse_ = args.reverse  # False or True
        feature_removal_level = args.feature_removal_level  # possible values: 'sample', 'set'

        train_gru_d(num_runs, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, n_epochs,
                    batch_size, upsampling_batch, split_type, feature_removal_level, missing_ratio, args.dataset, args.predictive_label)
