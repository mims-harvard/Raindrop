"""
Adapted from: https://github.com/mlds-lab/interp-net
Works with Tensorflow 1
"""

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import argparse
import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # todo: now makes CUDA invisible
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auc_score
import keras
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, GRU, Lambda, Permute
from keras.models import Model
from interpolation_layer import single_channel_interp, cross_channel_interp
# from mimic_preprocessing import load_data, trim_los, fix_input_format
import warnings
warnings.filterwarnings("ignore")


def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(perc*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(perc*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


def mean_imputation(vitals, mask):
    """For the time series missing entirely, our interpolation network 
    assigns the starting point (time t=0) value of the time series to 
    the global mean before applying the two-layer interpolation network.
    In such cases, the first interpolation layer just outputs the global
    mean for that channel, but the second interpolation layer performs 
    a more meaningful interpolation using the learned correlations from
    other channels."""
    counts = np.sum(np.sum(mask, axis=2), axis=0)
    mean_values = np.sum(np.sum(vitals*mask, axis=2), axis=0)/counts
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i, j]) == 0:
                mask[i, j, 0] = 1
                vitals[i, j, 0] = mean_values[j]
    return


# interpolation-prediction network
def interp_net():
    if gpu_num > 1:
        dev = "/cpu:0"
    else:
        dev = "/gpu:0"

    dev = "/cpu:0"  # todo: now always uses CPU

    with tf.device(dev):
        main_input = Input(shape=(4*num_features, timestamp), name='input')
        sci = single_channel_interp(ref_points, hours_look_ahead)
        cci = cross_channel_interp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True), reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh', recurrent_dropout=0.0, dropout=0.0)(z)
        main_output = Dense(2, activation='softmax', name='main_output')(z)
        orig_model = Model([main_input], [main_output, aux_output])

    if gpu_num > 1:
        model = multi_gpu_model(orig_model, gpus=gpu_num)
    else:
        model = orig_model
    print(orig_model.summary())
    return model


def customloss(ytrue, ypred):
    """ Autoencoder loss
    """
    wc = np.load('../data/' + dataname + '_std.npy').T
    y = ytrue[:, :num_features, :]
    m2 = ytrue[:, 3*num_features:4*num_features, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_features:2*num_features, :]
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    x = x/(wc**2)  # dividing by standard deviation
    x = tf.reduce_sum(x, axis=1)/num_features
    return tf.reduce_mean(x)


def random_sample(idx_0, idx_1, batch_size):
    """
    Returns a balanced sample by randomly sampling without replacement.

    :param idx_0: indices of negative samples
    :param idx_1: indices of positive samples
    :param batch_size: batch size
    :return: indices of balanced batch of negative and positive samples
    """
    idx0_batch = np.random.choice(idx_0, size=int(batch_size / 2), replace=False)
    idx1_batch = np.random.choice(idx_1, size=int(batch_size / 2), replace=False)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default='P19', choices=['P12', 'P19'],
                help="dataset to use")
ap.add_argument("-g", "--gpus", type=int, default=1,
                help="# of GPUs to use for training")
ap.add_argument("-batch", "--batch_size", type=int, default=256,
                help="# batch size to use for training")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="# of epochs for training")
ap.add_argument("-ref", "--reference_points", type=int,
                default=192, help="# of reference points")
ap.add_argument("-units", "--hidden_units", type=int,
                default=100, help="# of hidden units")
ap.add_argument("-hfadm", "--hours_from_adm", type=int,
                default=48, help="Hours of record to look at")

args = vars(ap.parse_args())
gpu_num = args["gpus"]
epoch = args["epochs"]
hid = args["hidden_units"]
ref_points = args["reference_points"]
hours_look_ahead = args["hours_from_adm"]
batch_size = args["batch_size"]

# Loading dataset
# y : (N,) discrete for classification, real values for regression
# x : (N, D, tn) input multivariate time series data with dimension
#     where N is number of data cases, D is the dimension of
#     sparse and irregularly sampled time series and tn is the union
#     of observed time stamps in all the dimension for a data case n.
#     Since each tn is of variable length, we pad them with zeros to
#     have an array representation.
# m : (N, D, tn) where m[i,j,k] = 0 means that x[i,j,k] is not observed.
# T : (N, D, tn) represents the actual time stamps of observation;

dataname = args["dataset"]
y_train = np.int64(np.load('../data/' + dataname + '_y_train.npy'))
y_test = np.int64(np.load('../data/' + dataname + '_y_test.npy'))


'''
normalized = False

if normalized:
    X_train = np.load('../data/' + dataname + '_X_train.npy')
else:
    X_train = np.load('../data/' + dataname + '_X_train_not_normalized.npy')
X_time_train = np.load('../data/' + dataname + '_X_time_train.npy')

if normalized:
    X_test = np.load('../data/' + dataname + '_X_test.npy')
else:
    X_test = np.load('../data/' + dataname + '_X_test_not_normalized.npy')
X_time_test = np.load('../data/' + dataname + '_X_time_test.npy')

X_train = np.transpose(X_train, (1, 2, 0))
X_test = np.transpose(X_test, (1, 2, 0))
X_time_train = np.transpose(X_time_train, (1, 0))
X_time_test = np.transpose(X_time_test, (1, 0))

num_attributes = X_train.shape[1]
X_time_train = np.repeat(X_time_train[:, np.newaxis, :], num_attributes, axis=1)
X_time_test = np.repeat(X_time_test[:, np.newaxis, :], num_attributes, axis=1)

X_train_mask = (X_train != 0).astype(int)
X_test_mask = (X_test != 0).astype(int)

# mean imputation and input format -- todo
mean_imputation(X_train, X_train_mask)
mean_imputation(X_test, X_test_mask)
X_train = np.concatenate((X_train, X_train_mask, X_time_train, hold_out(X_train_mask)), axis=1)
X_test = np.concatenate((X_test, X_test_mask, X_time_test, hold_out(X_test_mask)), axis=1)

epsilon = 1e-8
X_train = np.absolute(X_train) + epsilon
X_test = np.absolute(X_test) + epsilon

np.save('P12_X_train_all.npy', X_train)
np.save('P12_X_test_all.npy', X_test)
'''

X_train = np.load(dataname + '_X_train_all.npy')
X_test = np.load(dataname + '_X_test_all.npy')

timestamp = X_train.shape[2]
num_features = X_train.shape[1] // 4

upsampling_batch = True
if upsampling_batch:
    train_data_upsamled_X = []
    train_data_upsamled_y = []
    true_labels = y_train
    idx_0 = np.where(true_labels == 0)[0]
    idx_1 = np.where(true_labels == 1)[0]
    for _ in range(len(true_labels) // batch_size):
        indices = random_sample(idx_0, idx_1, batch_size)
        for i in indices:
            train_data_upsamled_X.append(X_train[i])
            train_data_upsamled_y.append(y_train[i])
    X_train = np.array(train_data_upsamled_X)
    y_train = np.array(train_data_upsamled_y)

results = {}
results['loss'] = []
results['auc'] = []
results['acc'] = []
results['auprc'] = []

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
callbacks_list = [earlystop, reduce_lr]

num_runs = 5
for i in range(num_runs):
    print("Run", i+1)

    model = interp_net()  # re-initializing every time
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'main_output': tf.keras.losses.SparseCategoricalCrossentropy(), 'aux_output': customloss},
        loss_weights={'main_output': 1., 'aux_output': 0.001},
        metrics={'main_output': 'accuracy'})

    model.fit(
        {'input': X_train}, {'main_output': y_train, 'aux_output': X_train},
        batch_size=batch_size,
        callbacks=callbacks_list,
        nb_epoch=epoch,
        validation_split=0.1,
        verbose=2,
        shuffle=True)

    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = y_pred[0]

    total_loss, score, reconst_loss, acc = model.evaluate(
        {'input': X_test},
        {'main_output': y_test, 'aux_output': X_test},
        batch_size=batch_size,
        verbose=0)

    results['loss'].append(score)
    results['acc'].append(acc)
    results['auc'].append(auc_score(y_test, y_pred[:, 1]))
    results['auprc'].append(auprc(y_test, y_pred[:, 1]))
    print('\n\n', results, '\n\n')
    i += 1

aucs = np.array(results['auc']) * 100
auprcs = np.array(results['auprc']) * 100
mean_auc, std_auc = np.mean(aucs), np.std(aucs)
mean_auprc, std_auprc = np.mean(auprcs), np.std(auprcs)
print(dataname)
print('AUROC: %.2f +/- %.2f' % (mean_auc, std_auc))
print('AUPRC: %.2f +/- %.2f' % (mean_auprc, std_auprc))


