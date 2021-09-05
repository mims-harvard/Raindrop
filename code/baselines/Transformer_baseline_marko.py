"""
Transformer baseline for P12 mortality data.
"""
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_baselines import *
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from models import Transformer_P12, TransformerModel2


def number_parameters(model):
    """
    Print the number of model's parameters (trainable and all).

    :param model: model instance
    :return: None
    """
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: ', f'{total_parameters:,}')
    print('Trainable parameters: ', f'{trainable_parameters:,}')


def positional_encoding(X_features, d_model, max_len, X_time):
    """
    Function for positional encoding from the paper "Attention is all you need".

    :param X_features: time series features for all samples
    :param d_model: dimension of a single sequential instance (number of parameters per instance)
    :param max_len: maximum length of the sequence (time series)
    :param X_time: times, when observations were measured; size = (batch, seq_len, 1)
    :return: X_feature with positional encoding (retruned as tensor)
    """
    X_features = torch.from_numpy(X_features)
    for i, sample in enumerate(X_features):
        pe = torch.zeros(max_len, d_model)
        positions = X_time[i]
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(positions / div_term)
        pe[:, 1::2] = torch.cos(positions / div_term)
        X_features[i] += pe
    return X_features


class PositionalEncoding(nn.Module):
    """
    Implement positional encoding following the paper "Attention is all you need" for P12 dataset.
    """
    def __init__(self, d_model, max_len, dropout=0.1):
        """
        Initialize the instance - prepare PE matrix and set dropout rate.

        :param d_model: dimension of a single sequential instance (number of parameters per instance)
        :param max_len: maximum length of the sequence (time series)
        :param dropout: dropout rate
        """
        super().__init__()

        # prepare PE matrix
        self.pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # here we lose the notion of exact time observations
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        self.pe[:, 0::2] = torch.sin(positions / div_term)
        self.pe[:, 1::2] = torch.cos(positions / div_term)

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_enc', self.pe)

    def forward(self, x):
        x = x + self.pe   # concatenate input with positional encodings
        return self.dropout(x)



def train_test_model(d_model, max_len, n_heads, dim_feedforward, X_train, X_val, X_test, num_runs, num_epochs,
                     learning_rate, dropout, batch_size, upsampling_factor=None, upsampling_batch=False):
    """
    Train the model.

    :param d_model: dimension of a single sequential instance (number of features per instance)
    :param max_len: maximum length of the sequence (time series)
    :param n_heads: number of attention heads
    :param dim_feedforward: dimension of the feedforward network model
    :param X_train: (X_features_train, X_static_train, X_time_train, y_train)
    :param X_val: (X_features_val, X_static_val, X_time_val, y_val)
    :param X_test: (X_features_test, X_static_test, X_time_test, y_test)
    :param num_runs: number of independent runs
    :param num_epochs: number of epochs
    :param learning_rate: learning rate for optimizer
    :param dropout: dropout rate
    :param batch_size: batch size
    :param upsampling_factor: upsampling of minority class by desired integer factor in train set, default=None (no upsampling)
    :param upsampling_batch: boolean to determine if number of positive and negative samples in each batch should be equal, default=False
    :return: None
    """
    X_features_train, X_static_train, X_time_train, y_train = X_train
    X_features_val, X_static_val, X_time_val, y_val = X_val
    X_features_test, X_static_test, X_time_test, y_test = X_test

    if upsampling_factor is not None:   # upsampling of minority class
        X_features_train, X_static_train, X_time_train, y_train = upsampling(X_train, upsampling_factor)
        print('\nSize after upsampling: ', X_features_train.shape, X_static_train.shape, X_time_train.shape, y_train.shape)
        pos_n = torch.count_nonzero(y_train)
        print('Positive samples: %d, negative samples: %d\n' % (pos_n, len(y_train) - pos_n))

    acc_all = []
    auc_all = []
    aupr_all = []
    model_path = './saved/best_model.pt'

    for r in range(num_runs):
        model = Transformer_P12(d_model, max_len, n_heads, dim_feedforward, dropout=dropout)

        # model = TransformerModel2(d_inp, d_model, n_heads, dim_feedforward, nlayers, dropout, max_len,
        #                           d_static, MAX, 0.5, aggreg, n_classes)


        model = model.double()
        if r == 0:
            print(model)
            number_parameters(model)

        loss_fun = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=1, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

        print('\n------------------\nRUN %d: Training started\n------------------' % r)

        best_aupr_val = 0
        for epoch in range(num_epochs):
            model.train()   # set model to the training mode

            permuted_idx = torch.randperm(len(X_features_train))

            for i in range(0, len(X_features_train), batch_size):
                # zero out all of the gradients
                optimizer.zero_grad()

                # make batches of samples
                if upsampling_batch:    # the same number of positive and negative samples in a batch
                    idx_0 = np.where(y_train.detach().cpu().numpy() == 0)[0]
                    idx_1 = np.where(y_train.detach().cpu().numpy() == 1)[0]
                    indices = random_sample(idx_0, idx_1, batch_size)
                else:   # batch distribution is the same as overall distribution
                    indices = permuted_idx[i:i + batch_size]

                batch_X_features, batch_X_static, batch_X_time, batch_y = \
                    X_features_train[indices], X_static_train[indices], X_time_train[indices], y_train[indices]

                # forward pass
                y_pred = model(batch_X_features, batch_X_static, batch_X_time)

                # compute and print loss
                loss = loss_fun(y_pred, batch_y)

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # make an update to model parameters
                optimizer.step()

            # print('Epoch %d: training loss: %.3f' % (epoch, loss.item()))

            # check validation set
            model.eval()    # set model to the evaluation mode
            if epoch % 1 == 0:
                with torch.no_grad():
                    y_pred = np.zeros(shape=(len(X_features_val), 2))
                    for i in range(0, len(X_features_val), batch_size):
                        y_pred[i:i + batch_size, :] = model(X_features_val[i:i + batch_size], X_static_val[i:i + batch_size],
                                                            X_time_val[i:i + batch_size]).detach().cpu().numpy()
                    y_pred = torch.from_numpy(y_pred)

                    val_loss = loss_fun(y_pred, y_val)
                    y_pred = torch.squeeze(nn.functional.softmax(y_pred, dim=1))

                    # compute classification accuracy
                    acc_val = np.sum(np.array(y_val) == np.argmax(np.array(y_pred), axis=1)) / y_val.shape[0]

                    # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
                    auc_val = roc_auc_score(y_val, y_pred[:, 1])

                    # compute average precision and area under precision-recall curve
                    aupr_val = average_precision_score(y_val, y_pred[:, 1])
                    scheduler.step(aupr_val)    # reduce learning rate when this metric has stopped improving

                    print('Non-zero predictions = ', np.count_nonzero(np.argmax(np.array(y_pred), axis=1)))
                    print("VALIDATION: Epoch %d, val_acc: %.2f, val_loss: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                          (epoch, acc_val * 100, val_loss.item(), aupr_val * 100, auc_val * 100))

                    # save the best model based on 'aupr'
                    if aupr_val > best_aupr_val:
                        best_aupr_val = aupr_val
                        torch.save(model, model_path)

                    print(confusion_matrix(y_val, np.argmax(y_pred, axis=1), labels=[0, 1]))

        print('\n------------------\nRUN %d: Training finished\n------------------' % r)

        # use the best model on validation set to predict on test set
        model = torch.load(model_path)
        model.eval()

        with torch.no_grad():
            y_pred = np.zeros(shape=(len(X_features_test), 2))
            for i in range(0, len(X_features_test), batch_size):
                y_pred[i:i + batch_size, :] = model(X_features_test[i:i + batch_size], X_static_test[i:i + batch_size],
                                                    X_time_test[i:i + batch_size]).detach().cpu().numpy()
            y_pred = torch.from_numpy(y_pred)

            y_pred = torch.squeeze(nn.functional.softmax(y_pred, dim=1))

            # compute classification accuracy
            acc_test = np.sum(np.array(y_test) == np.argmax(np.array(y_pred), axis=1)) / y_test.shape[0]

            # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
            auc_test = roc_auc_score(y_test, y_pred[:, 1])

            # compute average precision and area under precision-recall curve
            aupr_test = average_precision_score(y_test, y_pred[:, 1])

            print("\nTEST: test_acc: %.2f aupr_test: %.2f, auc_test: %.2f" %
                  (acc_test * 100, aupr_test * 100, auc_test * 100))

            print(confusion_matrix(y_test, np.argmax(y_pred, axis=1), labels=[0, 1]))

            acc_all.append(acc_test * 100)
            auc_all.append(auc_test * 100)
            aupr_all.append(aupr_test * 100)

    # print mean and std of all metrics
    acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
    mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
    mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
    mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
    print('------------------------------------------')
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))


if __name__ == '__main__':
    # missing_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # ratios [0, 1] of missing variables in validation and test set
    missing_ratios = [0.2]
    for missing_ratio in missing_ratios:
        base_path = '../../P12data'
        split_idx = 1
        split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'

        normalization = False
        imputation_method = None  # possible values: None, 'mean', 'forward', 'kNN', 'MICE' (slow execution), 'CubicSpline'
        split_type = 'random'   # possible values: 'random', 'age', 'gender'
        feature_removal_level = 'set'   # possible values: 'sample', 'set'
        # missing_ratio = 0.0     # ratio [0, 1] of missing variables in validation and test set

        (X_features_train, X_static_train, X_time_train, y_train), (X_features_val, X_static_val, X_time_val, y_val), (X_features_test, X_static_test, X_time_test, y_test) = \
            read_and_prepare_data(base_path, split_path, normalization, feature_removal_level, missing_ratio, imputation=imputation_method, split_type=split_type)

        d_model = 36    # number of features per time step
        max_len = 215   # max length of time series
        n_heads = 8     # number of heads does not change the number of model parameters
        dim_feedforward = 64

        # apply positional encoding to the input
        X_features_train = positional_encoding(X_features_train, d_model, max_len, X_time_train)
        X_features_val = positional_encoding(X_features_val, d_model, max_len, X_time_val)
        X_features_test = positional_encoding(X_features_test, d_model, max_len, X_time_test)
        print(X_features_train.shape, X_features_val.shape, X_features_test.shape)

        num_runs = 5
        num_epochs = 20
        batch_size = 128
        learning_rate = 0.001
        dropout = 0.3
        upsampling_factor = None    # None if no whole set upsampling is desired
        upsampling_batch = True     # True if we want the same number of positive nad negative samples in each batch
        X_train = (X_features_train, X_static_train, X_time_train, y_train)
        X_val = (X_features_val, X_static_val, X_time_val, y_val)
        X_test = (X_features_test, X_static_test, X_time_test, y_test)

        train_test_model(d_model, max_len, n_heads, dim_feedforward, X_train, X_val, X_test, num_runs, num_epochs,
                         learning_rate, dropout, batch_size, upsampling_factor, upsampling_batch)
        print('\nAbove results for missing ratio: %d\n\n\n' % missing_ratio)

