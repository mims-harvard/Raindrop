import sys
sys.path.append('../')

import torch
import numpy as np
from sklearn.preprocessing import normalize

from imputations import *
from utils_phy12 import *


def read_and_prepare_data(base_path, split_path, normalization, imputation=None):
    """
    Read data from the disk and prepare it into train, validation and test set.

    :param base_path: base path to the data
    :param split_path: specific path to the data
    :param: normalization: boolean whether to normalize the data
    :param: imputation: imputation method for missing values, default: no imputation (zeroes for missing values),
                        possible values: 'mean', 'forward', 'kNN', 'MICE', 'CubicSpline'
    :return: list of (X_features, X_static, X_time, y) for train, validation and test set
    """
    # prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_split(base_path, split_path)
    # print(X_train.shape, len(X_val), len(X_test), y_train.shape, len(y_val), len(y_test))

    """
    patient_idx = 5
    print(X_train[patient_idx].keys())
    print(X_train[patient_idx]["id"])
    print(X_train[patient_idx]["static"])
    print(X_train[patient_idx]["extended_static"])
    print(X_train[patient_idx]["arr"].shape)
    print(X_train[patient_idx]["arr"][:3, :3])
    print(X_train[patient_idx]["time"].shape)
    print(X_train[patient_idx]["time"][:3])
    print(X_train[patient_idx]["length"])
    """

    X_features_train, X_static_train, X_time_train, y_train = prepare_3D_data(X_train, y_train)
    X_features_val, X_static_val, X_time_val, y_val = prepare_3D_data(X_val, y_val)
    X_features_test, X_static_test, X_time_test, y_test = prepare_3D_data(X_test, y_test)

    # get stats for time series features from training data
    features_means = get_features_mean(X_features_train)
    # print(features_means)

    # get stats for static features from training data
    static_means = get_static_mean(X_static_train)
    # print(static_means)

    # impute missing values
    if imputation == 'mean':
        X_features_train, X_static_train = mean_imputation(X_features_train, X_static_train, X_time_train, features_means, static_means)
        X_features_val, X_static_val = mean_imputation(X_features_val, X_static_val, X_time_val, features_means, static_means)
        X_features_test, X_static_test = mean_imputation(X_features_test, X_static_test, X_time_test, features_means, static_means)
    elif imputation == 'forward':
        X_features_train = forward_imputation(X_features_train, X_time_train)
        X_features_val = forward_imputation(X_features_val, X_time_val)
        X_features_test = forward_imputation(X_features_test, X_time_test)
    elif imputation == 'kNN':
        # X_features_train = kNN_imputation(X_features_train, X_time_train)
        # X_features_val = kNN_imputation(X_features_val, X_time_val)
        # X_features_test = kNN_imputation(X_features_test, X_time_test)

        # load saved files for speed-up
        X_features_train = np.load('X_features_train_kNN_imputed.npy')
        X_features_val = np.load('X_features_validation_kNN_imputed.npy')
        X_features_test = np.load('X_features_test_kNN_imputed.npy')
    elif imputation == 'MICE':  # might have problems with high memory usage and convergence
        X_features_train = MICE_imputation(X_features_train, X_time_train)
        X_features_val = MICE_imputation(X_features_val, X_time_val)
        X_features_test = MICE_imputation(X_features_test, X_time_test)
    elif imputation == 'CubicSpline':
        X_features_train = cubic_spline_imputation(X_features_train, X_time_train)
        X_features_val = cubic_spline_imputation(X_features_val, X_time_val)
        X_features_test = cubic_spline_imputation(X_features_test, X_time_test)

    # normalize data
    if normalization:
        # normalization of time series features
        X_features_train = normalize_ts(X_features_train)
        X_features_val = normalize_ts(X_features_val)
        X_features_test = normalize_ts(X_features_test)

        # normalization of static features
        X_static_train = normalize_static(X_static_train)
        X_static_val = normalize_static(X_static_val)
        X_static_test = normalize_static(X_static_test)

    # change numpy arrays to tensors (X_features will be changed later)
    X_static_train = torch.from_numpy(X_static_train)
    X_static_val = torch.from_numpy(X_static_val)
    X_static_test = torch.from_numpy(X_static_test)
    X_time_train = torch.from_numpy(X_time_train)
    X_time_val = torch.from_numpy(X_time_val)
    X_time_test = torch.from_numpy(X_time_test)

    return [(X_features_train, X_static_train, X_time_train, y_train),
            (X_features_val, X_static_val, X_time_val, y_val),
            (X_features_test, X_static_test, X_time_test, y_test)]


def prepare_3D_data(X, y):
    """
    Prepare data for the model input.

    :param X: list of dictionary values
    :param y: list of target values
    :return: X_features (3D), X_static (2D), X_time (3D), y (as tensor)
    """
    X_features = np.zeros((len(X), *X[0]['arr'].shape))  # shape: (batch, 215, 36)
    X_static = np.zeros((len(X), len(X[0]['extended_static'])))  # shape: (batch, 9)
    X_time = np.zeros((len(X), X[0]['time'].shape[0], 1))  # shape: (batch, 215, 1)
    for i in range(len(X)):
        X_features[i] = X[i]['arr']
        X_static[i] = X[i]['extended_static']
        X_time[i] = X[i]["time"]

    # # one-hot-encode target values (nn.CrossEntropyLoss() does not expect one-hot encoded targets)
    # y = torch.from_numpy(np.array(list(map(lambda x: [0, 1] if x == 0 else [1, 0], y))))  # shape: (batch, 2)

    # y to 1D tensor
    y = torch.squeeze(torch.from_numpy(y))

    print(X_features.shape, X_static.shape, X_time.shape, y.shape)
    return X_features, X_static, X_time, y


def get_features_mean(X_features):
    """
    Calculate means of all time series features (36 features in P12 dataset).

    :param X_features: time series features for all samples in training set
    :return: list of means for all features (36)
    """
    samples, timesteps, features = X_features.shape
    X = np.reshape(X_features, newshape=(samples*timesteps, features)).T
    means = []
    for row in X:
        row = row[row > 0]
        means.append(np.mean(row))
    return means


def get_static_mean(X_static):
    """
    Calculate means of static feature, which are not categorical.

    :param X_static: extended static features for all samples in training set
    :return: list of means for all non-categorical variables (Age, Height, Weight)
    """
    # Static features: ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
    bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    means = []
    for i in range(len(bool_categorical)):
        if bool_categorical[i] == 0:  # not categorical
            vals = X_static[:, i]
            vals = vals[vals > 0]
            means.append(np.mean(vals))
    return means


def normalize_ts(X_features):
    """
    Perform L2 normalization for each time series feature in all samples.

    :param X_features: time series features for all samples
    :return: X_features, normalized
    """
    for i, sample in enumerate(X_features):
        X_features[i] = normalize(sample, axis=0)
    return X_features


def normalize_static(X_static):
    """
    Perform L2 normalization for non-categorical static features in all samples.

    :param X_static: extended static features for all samples
    :return: X_static with normalized non-categorical features
    """
    # Static features: ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
    bool_categorical = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0])
    norm_idx = np.where(bool_categorical == 0)[0]

    X_static_numerical = X_static[:, norm_idx]
    X_static_numerical_normalized = normalize(X_static_numerical, axis=0)

    X_static[:, norm_idx] = X_static_numerical_normalized
    return X_static


if __name__ == '__main__':
    """
    P12 data (11988 samples):
    9590 training samples
    1199 validation samples
    1199 test samples
    
    Each sample has:
    "arr".shape = (215, 36)
    "time".shape = (215, 1)
    """
    base_path = '../../P12data'
    split_idx = 1
    split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'

    normalization = True
    imputation_method = None   # possible values: None, 'mean', 'forward', 'kNN', 'MICE', 'CubicSpline'

    (X_features_train, X_static_train, X_time_train, y_train), (X_features_val, X_static_val, X_time_val, y_val), (X_features_test, X_static_test, X_time_test, y_test) = read_and_prepare_data(base_path, split_path, normalization, imputation=imputation_method)





