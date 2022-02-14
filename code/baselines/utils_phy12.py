
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


def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True, dataset='P12', predictive_label='mortality'):
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
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

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
    elif split_type == 'age':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)

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

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    # extract mortality labels
    if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
        if predictive_label == 'mortality':
            y = arr_outcomes[:, -1].reshape((-1, 1))
        elif predictive_label == 'LoS':  # for P12 only
            y = arr_outcomes[:, 3].reshape((-1, 1))
            y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    elif dataset == 'eICU':
        y = arr_outcomes[..., np.newaxis]
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f], eps])
    return mf, stdf


def get_features_mean(X_features):
    """
    Calculate means of all time series features (36 features in P12 dataset).

    :param X_features: time series features for all samples in training set
    :return: list of means for all features
    """
    samples, timesteps, features = X_features.shape
    X = np.reshape(X_features, newshape=(samples*timesteps, features)).T
    means = []
    for row in X:
        row = row[row > 0]
        means.append(np.mean(row))
    return means


def mean_imputation(X_features, X_time, mean_features, missing_value_num):
    """
    Fill X_features missing values with mean values of all train samples.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :param mean_features: mean values of features from the training set
    :return: X_features, filled with mean values instead of zeros (missing observations)
    """
    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # check for inconsistency
    for i in range(len(X_features)):
        if np.any(X_features[i, time_length[i]:, :]):
            print('Inconsistency between X_features and X_time: features are measured without time stamp.')

    # impute times series features
    for i, sample in enumerate(X_features):
        X_features_relevant = sample[:time_length[i], :]
        missing_values_idx = np.where(X_features_relevant == missing_value_num)
        for row, col in zip(*missing_values_idx):
            X_features[i, row, col] = mean_features[col]

    return X_features


def forward_imputation(X_features, X_time, missing_value_num):
    """
    Fill X_features missing values with values, which are the same as its last measurement.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with last measurements instead of zeros (missing observations)
    """
    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            first_observation = True
            current_value = -1
            for k, observation in enumerate(ts[:time_length[i]]):
                if X_features[i, k, j] == missing_value_num and first_observation:
                    continue
                elif X_features[i, k, j] != missing_value_num:
                    current_value = X_features[i, k, j]
                    first_observation = False
                elif X_features[i, k, j] == missing_value_num and not first_observation:
                    X_features[i, k, j] = current_value

    return X_features


def cubic_spline_imputation(X_features, X_time, missing_value_num):
    """
    Fill X_features missing values with cubic spline interpolation.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with interpolated values
    """
    from scipy.interpolate import CubicSpline

    time_length = []
    for times in X_time:
        if np.where(times == missing_value_num)[0].size == 0:
            time_length.append(times.shape[0])
        elif np.where(times == missing_value_num)[0][0] == 0:
            time_length.append(np.where(times == missing_value_num)[0][1])
        else:
            time_length.append(np.where(times == missing_value_num)[0][0])

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            valid_ts = ts[:time_length[i]]
            zero_idx = np.where(valid_ts == missing_value_num)[0]
            non_zero_idx = np.nonzero(valid_ts)[0]
            y = valid_ts[non_zero_idx]

            if len(y) > 1:   # we need at least 2 observations to fit cubic spline
                x = X_time[i, :time_length[i], 0][non_zero_idx]
                x2interpolate = X_time[i, :time_length[i], 0][zero_idx]

                cs = CubicSpline(x, y)
                interpolated_ts = cs(x2interpolate)
                valid_ts[zero_idx] = interpolated_ts

                # set values before first measurement to the value of first measurement
                first_obs_index = non_zero_idx[0]
                valid_ts[:first_obs_index] = np.full(shape=first_obs_index, fill_value=valid_ts[first_obs_index])

                # set values after last measurement to the value of last measurement
                last_obs_index = non_zero_idx[-1]
                valid_ts[last_obs_index:] = np.full(shape=time_length[i] - last_obs_index, fill_value=valid_ts[last_obs_index])

                X_features[i, :time_length[i], j] = valid_ts

    return X_features


def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2,0,1)).reshape(F,-1)
    M = 1*(P_tensor>0) + 0*(P_tensor<=0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F,N,T)).transpose((1,2,0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

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

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, P_time, y_tensor


def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape

    P_time = np.zeros((len(P), T, 1))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor


def masked_softmax(A, epsilon=0.000000001):
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * (A != 0).float()
    A_softmax = A_exp / (torch.sum(A_exp, dim=0, keepdim=True) + epsilon)
    return A_softmax


def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def random_sample_8(ytrain, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(np.where(ytrain == 0)[0], size=int(B / 8), replace=replace)
    idx1_batch = np.random.choice(np.where(ytrain == 1)[0], size=int(B / 8), replace=replace)
    idx2_batch = np.random.choice(np.where(ytrain == 2)[0], size=int(B / 8), replace=replace)
    idx3_batch = np.random.choice(np.where(ytrain == 3)[0], size=int(B / 8), replace=replace)
    idx4_batch = np.random.choice(np.where(ytrain == 4)[0], size=int(B / 8), replace=replace)
    idx5_batch = np.random.choice(np.where(ytrain == 5)[0], size=int(B / 8), replace=replace)
    idx6_batch = np.random.choice(np.where(ytrain == 6)[0], size=int(B / 8), replace=replace)
    idx7_batch = np.random.choice(np.where(ytrain == 7)[0], size=int(B / 8), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch, idx2_batch, idx3_batch, idx4_batch, idx5_batch, idx6_batch, idx7_batch], axis=0)
    return idx


def evaluate(model, P_tensor, P_time_tensor, P_static_tensor, batch_size=100, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    T, N, Ff = P_tensor.shape

    n_batches, rem = N // batch_size, N % batch_size

    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        lengths = torch.sum(Ptime > 0, dim=0)
        out[start:start + batch_size] = model.forward(P, Pstatic, Ptime, lengths).detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        lengths = torch.sum(Ptime > 0, dim=0)
        out[start:start + rem] = model.forward(P, Pstatic, Ptime, lengths).detach().cpu()
    return out


def evaluate_standard(model, P_tensor, P_time_tensor, P_static_tensor, batch_size=100, n_classes=2, static=1):
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    lengths = torch.sum(P_time_tensor > 0, dim=0)
    out = model.forward(P_tensor, P_static_tensor, P_time_tensor, lengths)
    return out


def evaluate_MTGNN(model, P_tensor, P_static_tensor, static=1):
    P_tensor = P_tensor.cuda()

    P_tensor = torch.permute(P_tensor, (1, 0, 2))
    P_tensor = torch.unsqueeze(P_tensor, dim=1)
    P_tensor = P_tensor.transpose(2, 3)

    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    out = model.forward(P_tensor, P_static_tensor)
    return out


def evaluate_DGM2(model, P_tensor, P_static_tensor, static=1):
    # suppose P_time is equal in all patients
    P_time = torch.arange(P_tensor.size()[0])

    P_tensor = P_tensor.cuda()
    P_tensor = torch.permute(P_tensor, (1, 0, 2))

    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    out = model.forward(P_tensor, P_time, P_static_tensor)
    return out


def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res


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
