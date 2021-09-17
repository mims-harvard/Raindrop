# Utility functions
#
# Author: Theodoros Tsiligkaridis
# Last updated: April 26 2021
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True, dataset='P12'):
    # load data
    if dataset == 'P12':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'eICU':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'eICU_'

    #     print(len(Pdict_list), arr_outcomes.shape)

    # Pdict_list = np.load(base_path + '/PTdict_list.npy', allow_pickle=True)
    # arr_outcomes = np.load(base_path + '/arr_outcomes.npy', allow_pickle=True)

    show_statistics = False
    if show_statistics:
        idx_under_65 = []
        idx_over_65 = []

        idx_male = []
        idx_female = []

        # variables for statistics
        all_ages = []
        female_count = 0
        male_count = 0
        all_BMI = []

        X_static = np.zeros((len(Pdict_list), len(Pdict_list[0]['extended_static'])))
        for i in range(len(Pdict_list)):
            X_static[i] = Pdict_list[i]['extended_static']
            age, gender_0, gender_1, height, _, _, _, _, weight = X_static[i]
            if age > 0:
                all_ages.append(age)
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender_0 == 1:
                female_count += 1
                idx_female.append(i)
            if gender_1 == 1:
                male_count += 1
                idx_male.append(i)
            if height > 0 and weight > 0:
                all_BMI.append(weight / ((height / 100) ** 2))

        # plot statistics
        plt.hist(all_ages, bins=[i * 10 for i in range(12)])
        plt.xlabel('Years')
        plt.ylabel('# people')
        plt.title('Histogram of patients ages, age known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
                  (len(all_ages), np.mean(np.array(all_ages)), np.std(np.array(all_ages)), np.median(np.array(all_ages))))
        plt.show()

        plt.hist(all_BMI, bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        all_BMI = np.array(all_BMI)
        all_BMI = all_BMI[(all_BMI > 10) & (all_BMI < 65)]
        plt.xlabel('BMI')
        plt.ylabel('# people')
        plt.title('Histogram of patients BMI, height and weight known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
                  (len(all_BMI), np.mean(all_BMI), np.std(all_BMI), np.median(all_BMI)))
        plt.show()
        print('\nGender known: %d,  Male count: %d,  Female count: %d\n' % (male_count + female_count, male_count, female_count))

    # np.save('saved/idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
    # np.save('saved/idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
    # np.save('saved/idx_male.npy', np.array(idx_male), allow_pickle=True)
    # np.save('saved/idx_female.npy', np.array(idx_female), allow_pickle=True)

    # transformer_path = True
    if baseline==True:
        BL_path = ''
    else:
        BL_path = 'baselines/'

    if split_type == 'random':
        # load random indices from a split
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
        #     print(len(idx_train), len(idx_val), len(idx_test))
    elif split_type == 'age':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)

        # if transformer_path:    # relative path for for Transformer_baseline.py
        #     idx_train = np.load('saved/idx_under_65.npy', allow_pickle=True)
        #     idx_vt = np.load('saved/idx_over_65.npy', allow_pickle=True)
        # else:   # relative path for for set_function_baseline.py
        #     idx_train = np.load('baselines/saved/idx_under_65.npy', allow_pickle=True)
        #     idx_vt = np.load('baselines/saved/idx_over_65.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    elif split_type == 'gender':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)

        # if transformer_path:    # relative path for for Transformer_baseline.py
        #     idx_train = np.load('saved/idx_male.npy', allow_pickle=True)
        #     idx_vt = np.load('saved/idx_female.npy', allow_pickle=True)
        # else:   # relative path for for set_function_baseline.py
        #     idx_train = np.load('baselines/saved/idx_male.npy', allow_pickle=True)
        #     idx_vt = np.load('baselines/saved/idx_female.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    # extract mortality labels
    if dataset == 'P12' or dataset == 'P19':
        y = arr_outcomes[:, -1].reshape((-1, 1))
    elif dataset == 'eICU':
        y = arr_outcomes[..., np.newaxis]
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    # # check mortality rates in each set
    # mort_train = np.sum(ytrain)/len(ytrain)
    # mort_val   = np.sum(yval)/len(yval)
    # mort_test  = np.sum(ytest)/len(ytest)
    # print(mort_train, mort_val, mort_test)  # All around 0.14

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


# obtain mean, std statistics on train-set
def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    # find mean for each variable
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        # extract values on non-missing data
        vals_f = vals_f[vals_f > 0]
        # compute mean, std
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f], eps])
    return mf, stdf

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    # compute masking vectors
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    # input normalization
    # normalize by channel
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    # concatenate with mask M
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
#     Pfinal_tensor = Pnorm_tensor
    return Pfinal_tensor

# def mask_normalize(P_tensor, mf, stdf):
#     """ Normalize time series variables. Missing ones are set to zero after normalization. """
#     N, T, F = P_tensor.shape  # shape: (120, 215, 36)
#     Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)  # shape: (36, 25800)
#
#     # compute masking vectors
#     M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)  # M shape: (960, 215, 36).  M.sum() = 384791
#     M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
#
#     # input normalization
#     # Normalize by channel.
#     for f in range(F):
#         Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
#
#     """Why this step? This will remove all values smaller than average.
#     Before normalization, there are 384791 nonzero values (M.sum()),
#     After this step, there are only 162615 nonzero values
#     Why don't we use: Pf = Pf*M ?"""
#     # # set missing values to zero after normalization
#     # for f in range(F):
#     #     idx_missing = np.where(Pf[f, :] <= 0)
#     #     Pf[f, idx_missing] = 0
#
#     Pf = Pf * M_3D
#
#     Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
#     WW = 1 * (Pnorm_tensor > 0) + 0 * (Pnorm_tensor <= 0)
#     print(WW.sum())  # 162615,
#
#     # # concatenate with mask M
#     # Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
#
#     Pfinal_tensor = Pnorm_tensor
#
#     return Pfinal_tensor


# for static data
def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    # find mean for each static
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    # p12_ = np.load('../../P12data/processed_data/extended_static_params.npy')
    # p19_ = np.load('../../P19data/processed_data/labels_demogr.npy')
    # eicu_ = np.load('../../eICUdata/processed_data/eICU_static_vars.npy')

    if dataset == 'P12':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical[s] == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor


def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_static_tensor = np.zeros((len(P), D))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)


    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)

    y_tensor = y  # y is the mortality label
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)  # change type to LongTensor, shape: [960]
    return P_tensor, P_static_tensor, P_time, y_tensor

def masked_softmax(A, epsilon=0.000000001):
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * (A != 0).float()  # this step masks
    # A_softmax = A_exp / torch.sum(A_exp, dim=1, keepdim=True)
    A_softmax = A_exp / (torch.sum(A_exp, dim=0, keepdim=True) + epsilon) # softmax by column
    return A_softmax

def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def evaluate(model, P_tensor, P_time_tensor, P_static_tensor, batch_size=100, n_classes=2):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_static_tensor = P_static_tensor.cuda()

    T, N, Ff = P_tensor.shape
    N, Fs = P_static_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size

    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        Pstatic = P_static_tensor[start:start + batch_size]
        lengths = torch.sum(Ptime > 0, dim=0)
        out[start:start + batch_size] = model.forward(P, Pstatic, Ptime, lengths).detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        Pstatic = P_static_tensor[start:start + rem]
        lengths = torch.sum(Ptime > 0, dim=0)
        out[start:start + rem] = model.forward(P, Pstatic, Ptime, lengths).detach().cpu()
    return out

"""Xiang:"""
def evaluate_standard(model, P_tensor, P_time_tensor, P_static_tensor, batch_size=100, n_classes=2):
    # model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_static_tensor = P_static_tensor.cuda()

    lengths = torch.sum(P_time_tensor > 0, dim=0)
    out= model.forward(P_tensor, P_static_tensor, P_time_tensor, lengths)
    return out

# Adam using warmup
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
