import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

wandb = False

if wandb:
    import wandb

    # wandb.offline
    os.environ['WANDB_SILENT']="true"
    wandb.login(key=str('14734fe9c5574e019e8f517149a20d6fe1b2fd0d'))
    config = wandb.config
    # run = wandb.init(project='WBtest', config={'wandb_nb':'wandb_three_in_one_hm'})
    run = wandb.init(project='Raindrop', entity='XZ', config={'wandb_nb':'wandb_three_in_one_hm'})

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score
from models import ODEFunc, DiffeqSolver, GRU_unit_cluster, DGM2_O
from utils_phy12 import *

torch.manual_seed(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True, use female, older for training; if False, use female or younger for training')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
parser.add_argument('--imputation', type=str, default='no_imputation', choices=['no_imputation', 'mean', 'forward', 'cubic_spline'],
                    help='use this if you want to impute missing values')

args, unknown = parser.parse_known_args()


arch = 'standard'

model_path = '../../models/'

dataset = args.dataset
print('Dataset used: ', dataset)

if dataset == 'P12':
    base_path = '../../P12data'
elif dataset == 'P19':
    base_path = '../../P19data'
elif dataset == 'eICU':
    base_path = '../../eICUdata'
elif dataset == 'PAM':
    base_path = '../../PAMdata'


def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def one_hot_classes(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def create_net(n_inputs, n_outputs, n_layers=0, n_units=10, nonlinear=nn.Tanh, add_softmax=False, dropout=0.0):
    if n_layers >= 0:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Dropout(p=dropout))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    else:
        layers = [nn.Linear(n_inputs, n_outputs)]

        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)


feature_removal_level = args.feature_removal_level   # possible values: 'sample', 'set'
device = "cuda:0"

if args.withmissingratio == True:
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    missing_ratios = [0]

for missing_ratio in missing_ratios:
    num_epochs = 20
    learning_rate = 0.001

    if dataset == 'P12':
        d_static = 9
        d_inp = 36
        static_info = 1
    elif dataset == 'P19':
        d_static = 6
        d_inp = 34
        static_info = 1
    elif dataset == 'eICU':
        d_static = 399
        d_inp = 14
        static_info = 1
    elif dataset == 'PAM':
        d_static = 0
        d_inp = 17
        static_info = None

    d_model = 36
    nhid = 2 * d_model
    nlayers = 1
    nhead = 1

    dropout = 0.3

    if dataset == 'P12':
        max_len = 215
        n_classes = 2
    elif dataset == 'P19':
        max_len = 60
        n_classes = 2
    elif dataset == 'eICU':
        max_len = 300
        n_classes = 2
    elif dataset == 'PAM':
        max_len = 600
        n_classes = 8

    aggreg = 'mean'

    MAX = 100

    n_runs = 1
    n_splits = 5
    subset = False

    split = args.splittype
    reverse = args.reverse
    baseline = True

    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    precision_arr = np.zeros((n_splits, n_runs))
    recall_arr = np.zeros((n_splits, n_runs))
    F1_arr = np.zeros((n_splits, n_runs))
    for k in range(n_splits):
        split_idx = k + 1
        print('Split id: %d' % split_idx)

        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'

        # prepare the data:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split,
                                                                  reverse=reverse, baseline=baseline, dataset=dataset,
                                                                  predictive_label=args.predictive_label)
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        # impute missing values
        if args.imputation != 'no_imputation':
            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                X_features_train = np.array([d['arr'] for d in Ptrain])
                X_time_train = np.array([d['time'] for d in Ptrain])
                X_features_val = np.array([d['arr'] for d in Pval])
                X_time_val = np.array([d['time'] for d in Pval])
                X_features_test = np.array([d['arr'] for d in Ptest])
                X_time_test = np.array([d['time'] for d in Ptest])
            elif dataset == 'PAM':
                X_features_train = Ptrain
                X_time_train = np.array([np.arange(1, Ptrain.shape[1] + 1)[..., np.newaxis] for d in Ptrain])
                X_features_val = Pval
                X_time_val = np.array([np.arange(1, Pval.shape[1] + 1)[..., np.newaxis] for d in Pval])
                X_features_test = Ptest
                X_time_test = np.array([np.arange(1, Ptest.shape[1] + 1)[..., np.newaxis] for d in Ptest])

            if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
                missing_value_num = 0
            elif dataset == 'eICU':
                missing_value_num = -1

            if args.imputation == 'mean':
                features_means = get_features_mean(X_features_train)
                X_features_train = mean_imputation(X_features_train, X_time_train, features_means, missing_value_num)
                X_features_val = mean_imputation(X_features_val, X_time_val, features_means, missing_value_num)
                X_features_test = mean_imputation(X_features_test, X_time_test, features_means, missing_value_num)
            elif args.imputation == 'forward':
                X_features_train = forward_imputation(X_features_train, X_time_train, missing_value_num)
                X_features_val = forward_imputation(X_features_val, X_time_val, missing_value_num)
                X_features_test = forward_imputation(X_features_test, X_time_test, missing_value_num)
            elif args.imputation == 'cubic_spline':
                X_features_train = cubic_spline_imputation(X_features_train, X_time_train, missing_value_num)
                X_features_val = cubic_spline_imputation(X_features_val, X_time_val, missing_value_num)
                X_features_test = cubic_spline_imputation(X_features_test, X_time_test, missing_value_num)

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                for i, pat in enumerate(X_features_train):
                    Ptrain[i]['arr'] = pat
                for i, pat in enumerate(X_features_val):
                    Pval[i]['arr'] = pat
                for i, pat in enumerate(X_features_test):
                    Ptest[i]['arr'] = pat
            elif dataset == 'PAM':
                for i, pat in enumerate(X_features_train):
                    Ptrain[i] = pat
                for i, pat in enumerate(X_features_val):
                    Pval[i] = pat
                for i, pat in enumerate(X_features_test):
                    Ptest[i] = pat

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])

            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain,
                                                                                                         mf,
                                                                                                         stdf, ms, ss)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf,
                                                                                                 ms, ss)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf,
                                                                                                     stdf, ms,
                                                                                                     ss)

            print(Ptrain_tensor.shape, Ptrain_static_tensor.shape, Ptrain_time_tensor.shape, ytrain_tensor.shape)
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)

        # remove part of variables in validation and test set
        if missing_ratio > 0:
            num_all_features = Pval_tensor.shape[2]
            num_missing_features = round(missing_ratio * num_all_features)
            if feature_removal_level == 'sample':
                for i, patient in enumerate(Pval_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # values
                    Pval_tensor[i] = patient
                for i, patient in enumerate(Ptest_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)   # values
                    Ptest_tensor[i] = patient
            elif feature_removal_level == 'set':
                density_score_indices = np.load('saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                idx = density_score_indices[:num_missing_features].astype(int)
                Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
                Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)

        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        Pval_tensor = Pval_tensor.permute(1, 0, 2)
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

        for m in range(n_runs):
            print('- - Run %d - -' % (m + 1))

            num_nodes = Ptrain_tensor.size()[2]

            rec_ode_func = ODEFunc(
                input_dim=10,
                latent_dim=10,
                ode_func_net=create_net(10, 10),
                device=device).to(device)

            z0_diffeq_solver = DiffeqSolver(10, rec_ode_func, "euler", 10, odeint_rtol=1e-3, odeint_atol=1e-4,
                                            device=device)

            gru_update = GRU_unit_cluster(10, num_nodes, n_units=10, device=device, use_mask=False, dropout=0.0)

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                model = DGM2_O(10, num_nodes, 20, z0_diffeq_solver, z0_dim=10, n_gru_units=10, GRU_update=gru_update,
                               device=device, use_mask=False, dropout=0.0, use_static=True,
                               num_time_steps_and_static=(Ptrain_tensor.size()[0], Ptrain_static_tensor.size()[1]))

            elif dataset == 'PAM':
                model = DGM2_O(10, num_nodes, 20, z0_diffeq_solver, z0_dim=10, n_gru_units=10, GRU_update=gru_update,
                               device=device, use_mask=False, dropout=0.0, use_static=False,
                               num_time_steps_and_static=(Ptrain_tensor.size()[0], 0))

            model = model.cuda()

            criterion = torch.nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                   patience=1, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                strategy = 2
            elif dataset == 'PAM':
                strategy = 3

            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            batch_size = 128
            if strategy == 1:
                n_batches = 10
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])
            elif strategy == 3:
                n_batches = 30

            best_aupr_val = best_auc_val = 0.0

            start = time.time()
            if wandb:
                wandb.watch(model)
            for epoch in range(num_epochs):
                model.train()

                if strategy == 2:
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                for n in range(n_batches):
                    if strategy == 1:
                        idx = random_sample(idx_0, idx_1, batch_size)
                    elif strategy == 2:
                        """In each batch=128, 64 positive samples, 64 negative samples"""
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    elif strategy == 3:
                        idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                        # idx = random_sample_8(ytrain, batch_size)   # to balance dataset

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               None, ytrain_tensor[idx].cuda()

                    lengths = torch.sum(Ptime > 0, dim=0)

                    outputs = evaluate_DGM2(model, P, Pstatic, static=static_info)

                    optimizer.zero_grad()
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    train_probs = torch.squeeze(torch.sigmoid(outputs))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()
                    train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                    train_auprc = average_precision_score(train_y, train_probs[:, 1])
                elif dataset == 'PAM':
                    train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()
                    train_auroc = roc_auc_score(one_hot_classes(train_y, n_classes), train_probs)
                    train_auprc = average_precision_score(one_hot_classes(train_y, n_classes), train_probs)

                if wandb:
                    wandb.log({ "train_loss": loss.item(), "train_auprc": train_auprc, "train_auroc": train_auroc})

                """Validation"""
                model.eval()
                if epoch ==0 or epoch % 1 == 0:
                    with torch.no_grad():
                        # out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor, static=static_info)

                        n_batches = math.ceil(Pval_tensor.size()[1] / batch_size)

                        out_val_tensors = []
                        for n in range(n_batches):
                            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                                out_val_tensors.append(evaluate_DGM2(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :],
                                                   Pval_static_tensor[n * batch_size: (n + 1) * batch_size, :], static=static_info))
                            elif dataset == 'PAM':
                                out_val_tensors.append(evaluate_DGM2(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :],
                                                                      None,static=static_info))

                        out_val = torch.cat(out_val_tensors, dim=0)

                        out_val = torch.squeeze(torch.sigmoid(out_val))
                        out_val = out_val.detach().cpu().numpy()

                        val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                            auc_val = roc_auc_score(yval, out_val[:, 1])
                            aupr_val = average_precision_score(yval, out_val[:, 1])
                        elif dataset == 'PAM':
                            yval_int = np.int64(np.reshape(yval, (yval.shape[0], )))
                            auc_val = roc_auc_score(one_hot_classes(yval_int, n_classes), out_val)
                            aupr_val = average_precision_score(one_hot_classes(yval_int, n_classes), out_val)

                        print("Validation: Epoch %d,  val_loss:%.4f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                          val_loss.item(), aupr_val * 100, auc_val * 100))

                        if wandb:
                            wandb.log({ "val_loss": val_loss.item(), "val_auprc": aupr_val, "val_auroc": auc_val})

                        scheduler.step(aupr_val)
                        if auc_val > best_auc_val:
                            best_auc_val = auc_val
                            print(
                                "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (epoch, aupr_val * 100, auc_val * 100))
                            torch.save(model.state_dict(), model_path + arch + '_' + str(split_idx) + '.pt')

            end = time.time()
            time_elapsed = end - start
            print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

            """Testing"""
            model.load_state_dict(torch.load(model_path + arch + '_' + str(split_idx) + '.pt'))
            model.eval()

            with torch.no_grad():
                n_batches = math.ceil(Ptest_tensor.size()[1] / batch_size)

                out_test_tensors = []
                for n in range(n_batches):
                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                        out_test_tensors.append(evaluate_DGM2(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :],
                                                Ptest_static_tensor[n * batch_size: (n + 1) * batch_size, :], static=static_info))
                    elif dataset == 'PAM':
                        out_test_tensors.append(evaluate_DGM2(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :],
                                                               None, static=static_info))
                out_test = np.array(torch.cat(out_test_tensors, dim=0).detach().cpu())

                # out_test = out_test.detach().cpu()

                ypred = np.argmax(out_test, axis=1)
                denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
                probs = np.exp(out_test) / denoms
                probs = np.nan_to_num(probs, nan=0.0)

                acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                    auc = roc_auc_score(ytest, probs[:, 1])
                    aupr = average_precision_score(ytest, probs[:, 1])
                elif dataset == 'PAM':
                    ytest_int = np.int64(np.reshape(ytest, (ytest.shape[0],)))
                    auc = roc_auc_score(one_hot_classes(ytest_int, n_classes), probs)
                    aupr = average_precision_score(one_hot_classes(ytest_int, n_classes), probs)
                    precision = precision_score(ytest, ypred, average='macro', )
                    recall = recall_score(ytest, ypred, average='macro', )
                    F1 = f1_score(ytest, ypred, average='macro', )
                    print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (precision * 100, recall * 100, F1 * 100))

                print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
                print('classification report', classification_report(ytest, ypred))
                print(confusion_matrix(ytest, ypred, labels=list(range(n_classes))))

            # store
            acc_arr[k, m] = acc * 100
            auprc_arr[k, m] = aupr * 100
            auroc_arr[k, m] = auc * 100
            if dataset == 'PAM':
                precision_arr[k, m] = precision * 100
                recall_arr[k, m] = recall * 100
                F1_arr[k, m] = F1 * 100

    # pick best performer for each split based on max AUPRC
    idx_max = np.argmax(auprc_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
    if dataset == 'PAM':
        precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
        recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
        F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]

    print("missing ratio:{}, split type:{}, reverse:{}, using baseline:{}".format(missing_ratio, split, reverse, baseline))

    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    print('------------------------------------------')
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))
    if dataset == 'PAM':
        mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
        mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
        mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
        print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
        print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
        print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))

    if wandb:
        wandb.finish()

