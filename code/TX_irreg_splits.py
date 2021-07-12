# Irregular sample Transformer model - PhysioNet-2012
# Aggregation over 5 splits
#
# Author: Theodoros Tsiligkaridis
# Last updated: May 19 2021
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from Transformer import TransformerModel, TransformerModel2

from utils_phy12 import *

torch.manual_seed(1)

# training modes
arch = 'tx_irreg'

base_path = '../P12data'
model_path = './models/'

### show the names of variables and statistic descriptors
ts_params = np.load(base_path + '/processed_data/ts_params.npy', allow_pickle=True)
extended_static_params = np.load(base_path + '/processed_data/extended_static_params.npy', allow_pickle=True)
print('ts_params: ', ts_params)
print('extended_static_params: ', extended_static_params)

# training/model params
num_epochs = 20
learning_rate = 0.001

d_static = 9
# emb_len     = 10

d_inp = 36 * 2
# d_inp = 36*1

d_model = 256
nhid = 2 * d_model
# nhid = 256
# nhid = 512  # seems to work better than 2*d_model=256
# nhid = 1024
nlayers = 2
# nhead = 16 # seems to work better
nhead = 32
dropout = 0.3

max_len = 215

aggreg = 'mean'
# aggreg = 'max'

n_classes = 2
# MAX = d_model
MAX = 100

n_runs = 1  # change this from 1 to 1, in order to save CUDA memory.
n_splits = 1  # change this from 5 to 1, in order to save CUDA memory.

acc_arr = np.zeros((n_splits, n_runs))
auprc_arr = np.zeros((n_splits, n_runs))
auroc_arr = np.zeros((n_splits, n_runs))
for k in range(n_splits):
    split_idx = k + 1
    print('Split id: %d' % split_idx)
    split_path = '/splits/phy12_split' + str(split_idx) + '.npy'

    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path)

    T, F = Ptrain[0]['arr'].shape
    D = len(Ptrain[0]['extended_static'])

    # get mean, std stats from train set
    Ptrain_tensor = np.zeros((len(Ptrain), T, F))
    Ptrain_static_tensor = np.zeros((len(Ptrain), D))
    for i in range(len(Ptrain)):
        Ptrain_tensor[i] = Ptrain[i]['arr']
        Ptrain_static_tensor[i] = Ptrain[i]['extended_static']
    mf, stdf = getStats(Ptrain_tensor)
    ms, ss = getStats_static(Ptrain_static_tensor)

    Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                 stdf, ms, ss)
    Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
    Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                             ss)

    # convert to (seq_len, batch, feats)
    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    # convert to (seq_len, batch)
    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    for m in range(n_runs):
        print('- - Run %d - -' % (m + 1))

        # instantiate model
        model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                  d_static, MAX, 0.5, aggreg, n_classes)
        #         model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
        #                                   d_static, MAX, 0.5, 'mean', n_classes)
        #         model = TransformerModel(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
        #                                   d_static, MAX, n_classes)
        model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=1, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

        idx_0 = np.where(ytrain == 0)[0]
        idx_1 = np.where(ytrain == 1)[0]

        strategy = 2

        # Strategy 2: permute randomly each index set at each epoch, and expand x3 minority set
        n0, n1 = len(idx_0), len(idx_1)
        expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
        expanded_n1 = len(expanded_idx_1)

        batch_size = 128  # balanced batch size
        if strategy == 1:
            n_batches = 10  # number of batches to process per epoch
        elif strategy == 2:
            K0 = n0 // int(batch_size / 2)
            K1 = expanded_n1 // int(batch_size / 2)
            n_batches = np.min([K0, K1])

        best_aupr_val = best_auc_val = 0.0
        print('Epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))

        #         optimizer = NoamOpt(d_model, 5.0, 500, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        start = time.time()
        for epoch in range(num_epochs):
            model.train()

            if strategy == 2:
                # random shuffling of expanded_idx_1, idx_0
                ep1 = np.random.permutation(expanded_n1)
                p0 = np.random.permutation(n0)
                I1 = expanded_idx_1[ep1]
                I0 = idx_0[p0]

            for n in range(n_batches):
                if strategy == 1:
                    idx = random_sample(idx_0, idx_1, batch_size)
                elif strategy == 2:
                    idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                    idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)

                P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                       Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                lengths = torch.sum(Ptime > 0, dim=0)

                outputs = model.forward(P, Pstatic, Ptime, lengths)

                optimizer.zero_grad()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            if epoch % 1 == 0:
                with torch.no_grad():
                    out_val = evaluate(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor).numpy()
                    denoms = np.sum(np.exp(out_val), axis=1).reshape((-1, 1))
                    probs = np.exp(out_val) / denoms
                    ypred = np.argmax(out_val, axis=1)
                    auc_val = roc_auc_score(yval, probs[:, 1])
                    aupr_val = average_precision_score(yval, probs[:, 1])

                    print("Epoch %d, aupr_val: %.4f, auc_val: %.4f" % (epoch, aupr_val * 100, auc_val * 100))

                    #                     scheduler.step(aupr_val)

                    # save model
                    if aupr_val > best_aupr_val:
                        best_aupr_val = aupr_val
                        #                     if auc_val > best_auc_val:
                        #                         best_auc_val = auc_val
                        print(
                            "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (epoch, aupr_val * 100, auc_val * 100))
                        torch.save(model.state_dict(), model_path + arch + '_' + str(split_idx) + '.pt')

            if epoch == 3:
                end = time.time()
                time_elapsed = end - start
                print('-- Estimated train time: %.3f mins --' % (time_elapsed / 60.0 / 4 * num_epochs))

        end = time.time()
        time_elapsed = end - start
        print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

        # test evaluation
        model.load_state_dict(torch.load(model_path + arch + '_' + str(split_idx) + '.pt'))
        model.eval()

        with torch.no_grad():
            out_test = evaluate(model, Ptest_tensor, Ptest_time_tensor, Ptest_static_tensor).numpy()
            ypred = np.argmax(out_test, axis=1)
            denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
            probs = np.exp(out_test) / denoms

            auc = roc_auc_score(ytest, probs[:, 1])
            aupr = average_precision_score(ytest, probs[:, 1])
            acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]
            print('Test: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))

        # store
        acc_arr[k, m] = acc * 100
        auprc_arr[k, m] = aupr * 100
        auroc_arr[k, m] = auc * 100

# pick best performer for each split based on max AUPRC
idx_max = np.argmax(auprc_arr, axis=1)
acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]

# display mean and standard deviation
mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
print('------------------------------------------')
print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))

# save in numpy file
np.save('./results/' + arch + '_phy12.npy', [acc_vec, auprc_vec, auroc_vec])


