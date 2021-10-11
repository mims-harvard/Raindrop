import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from scipy.interpolate import CubicSpline


def mean_imputation(X_features, X_static, X_time, mean_features, mean_static):
    """
    Fill X_features and X_static missing values with mean values of all train samples.

    :param X_features: time series features for all samples
    :param X_static: extended static features for all samples
    :param X_time: times, when observations were measured
    :param mean_features: mean values of features from the training set
    :param mean_static: mean values of static features from the training set
    :return: X_features and X_static, filled with mean values instead of zeros (missing observations)
    """
    time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for times in X_time]

    # check for inconsistency
    for i in range(len(X_features)):
        if np.any(X_features[i, time_length[i]:, :]):
            print('Inconsistency between X_features and X_time: features are measured without time stamp.')

    # impute times series features
    for i, sample in enumerate(X_features):
        X_features_relevant = sample[:time_length[i], :]
        missing_values_idx = np.where(X_features_relevant == 0)
        for row, col in zip(*missing_values_idx):
            X_features[i, row, col] = mean_features[col]

    # impute static features
    missing_values_idx = np.where((X_static == 0) | (X_static == -1))
    for row, col in zip(*missing_values_idx):
        if col == 0:    # Age
            X_static[row, col] = mean_static[0]
        elif col == 3:    # Height
            X_static[row, col] = mean_static[1]
        elif col == 8:    # Weight
            X_static[row, col] = mean_static[2]

    return X_features, X_static


def forward_imputation(X_features, X_time):
    """
    Fill X_features missing values with values, which are the same as its last measurement.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with last measurements instead of zeros (missing observations)
    """
    time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for times in X_time]

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            first_observation = True
            current_value = -1
            for k, observation in enumerate(ts[:time_length[i]]):
                if X_features[i, k, j] == 0 and first_observation:
                    continue
                elif X_features[i, k, j] != 0:
                    current_value = X_features[i, k, j]
                    first_observation = False
                elif X_features[i, k, j] == 0 and not first_observation:
                    X_features[i, k, j] = current_value

    return X_features


def kNN_imputation(X_features, X_time):
    """
    Fill X_features missing values with mean values of k nearest neighbours.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with mean values of k nearest neighbours
    """
    time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for times in X_time]

    # impute times series features
    for i, sample in enumerate(X_features):
        X_features[i, :time_length[i], :][X_features[i, :time_length[i], :] == 0] = np.nan   # inside valid time put zeros to np.nan for KNNImputer
    X_features_2d = np.reshape(X_features, newshape=(X_features.shape[0], X_features.shape[1] * X_features.shape[2]))

    imputer = KNNImputer(n_neighbors=10, weights='uniform', metric='nan_euclidean', copy=False)
    imputer.fit_transform(X_features_2d)

    X_features = np.reshape(X_features_2d, newshape=X_features.shape)

    X_features = np.nan_to_num(X_features, nan=0)   # convert possible NaNs to 0

    # # save imputed data
    # np.save('saved/X_features_test_kNN_imputed', X_features)

    return X_features


def MICE_imputation(X_features, X_time):
    """
    Fill X_features missing values with MICE imputation.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with MICE imputation
    """
    time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for times in X_time]

    # impute times series features
    for i, sample in enumerate(X_features):
        X_features[i, :time_length[i], :][X_features[i, :time_length[i], :] == 0] = np.nan   # inside valid time put zeros to np.nan for IterativeImputer
    X_features_2d = np.reshape(X_features, newshape=(X_features.shape[0], X_features.shape[1] * X_features.shape[2]))

    imputer = IterativeImputer(n_nearest_features=10, skip_complete=True, min_value=0)
    imputer.fit_transform(X_features_2d)

    X_features = np.reshape(X_features_2d, newshape=X_features.shape)

    # # save imputed data
    # np.save('saved/X_features_train_MICE_imputed', X_features)

    return X_features


def cubic_spline_imputation(X_features, X_time):
    """
    Fill X_features missing values with cubic spline interpolation.

    :param X_features: time series features for all samples
    :param X_time: times, when observations were measured
    :return: X_features, filled with interpolated values
    """
    time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for times in X_time]

    # impute times series features
    for i, sample in enumerate(X_features):
        for j, ts in enumerate(sample.T):   # note the transposed matrix
            valid_ts = ts[:time_length[i]]
            zero_idx = np.where(valid_ts == 0)[0]
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

