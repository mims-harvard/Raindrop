# Raindrop strategy -- using PhysioNet-2012 dataset
#
# Author: Xiang Zhang
# Last updated: 2021

wandb = False


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

if wandb:
    import wandb
    os.environ['WANDB_SILENT']="true"

    wandb.login(key=str('14734fe9c5574e019e8f517149a20d6fe1b2fd0d'))
    config = wandb.config
    run = wandb.init(project='Raindrop', entity='xiang_zhang', config={'wandb_nb':'wandb_three_in_one_hm'})

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, accuracy_score
from models_rd import *
from utils_rd import *
# from utils_phy12 import *



from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from scipy.special import softmax


def generate_global_structure(data, K=10):
    observations = data[:, :, :36]
    cos_sim = torch.zeros([observations.shape[0], 36, 36])

    # overlap = torch.zeros([observations.shape[0], 36,36])
    for row in tqdm(range(observations.shape[0])):
        unit = observations[row].T
        cos_sim_unit = cosine_similarity(unit)  # shape: (9590, 9590)
        cos_sim[row] = torch.from_numpy(cos_sim_unit)

    ave_sim = torch.mean(cos_sim, dim=0)
    # Find the top K neighbors and softmax
    index = torch.argsort(ave_sim, dim=0)
    index_K = index < K  # K=10
    global_structure = index_K * ave_sim  #
    global_structure = masked_softmax(global_structure)  # softmax while mask out zero values
    return global_structure

def diffuse(unit, N=10):
    n_time = unit.shape[-1]
    keep = n_time//N  -1
    unit = unit[:, :keep*N].reshape([-1, keep, N])
    # return torch.mean(unit, dim=-1)
    return torch.max(unit, dim=-1).values


def generate_global_structure_diffuse(data, K=10, dataset ='P12'):
    if dataset == 'P12':
        n_features = 36
    elif dataset == 'P19':
        n_features = 34
    elif dataset == 'eICU':
        n_features = 14

    observations = Ptrain_tensor[:, :, :n_features]
    cos_sim = torch.zeros([observations.shape[0], n_features, n_features])

    # overlap = torch.zeros([observations.shape[0], n_features, n_features])
    for row in tqdm(range(observations.shape[0])):
        unit = observations[row].T  # unit.shape [36, 215]
        unit = diffuse(unit, N=10)  # diffuse the values into N-near following steps

        cos_sim_unit = cosine_similarity(unit)  # shape: (9590, 9590)
        cos_sim[row] = torch.from_numpy(cos_sim_unit)

    ave_sim = torch.mean(cos_sim, dim=0)
    # Find the top K neighbors and softmax
    index = torch.argsort(ave_sim, dim=0)
    index_K = index < K  # K=10
    global_structure = index_K * ave_sim  #
    global_structure = masked_softmax(global_structure)  # softmax while mask out zero values
    return global_structure

torch.manual_seed(1)

# training modes
arch = 'standard'


model_path = '../models/'

dataset = 'P12'     # possible values: 'P12', 'P19', 'eICU'
print('Dataset used: ', dataset)

if dataset == 'P12':
    base_path = '../P12data'
elif dataset == 'P19':
    base_path = '../P19data'
elif dataset == 'eICU':
    base_path = '../eICUdata'

# ### show the names of variables and statistic descriptors
# ts_params = np.load(base_path + '/processed_data/ts_params.npy', allow_pickle=True)
# extended_static_params = np.load(base_path + '/processed_data/extended_static_params.npy', allow_pickle=True)
# print('ts_params: ', ts_params)
# print('extended_static_params: ', extended_static_params)

# """Xiang"""
ts_params= ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine',
 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP',
 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2',
 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
 'Urine', 'WBC', 'pH']
extended_static_params=['Age', 'Gender=0', 'Gender=1', 'Height', 'ICUType=1', 'ICUType=2', 'ICUType=3',
 'ICUType=4', 'Weight']
"""setting split based on gender and age"""


"""split = 'random', 'age', 'gender"""
"""reverse= True: male, age<65 for training. 
 reverse=False: female, age>65 for training"""
"""baseline=True: run baselines. False: run our model (Raindrop)"""
split = 'random'
reverse = False
baseline = False  # Always False for Raindrop

feature_removal_level = 'sample'   # possible values: 'sample', 'set'

"""While missing_ratio >0, feature_removal_level is automatically used"""
# missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
missing_ratios = [0.8]
for missing_ratio in missing_ratios:
    # training/model params
    num_epochs = 10 #  20  # 20  # 30
    learning_rate = 0.0001

    d_static = 9
    if dataset == 'P12':
        d_static = 9
    elif dataset == 'P19':
        d_static = 6
    elif dataset == 'eICU':
        d_static = 399

    d_inp = 36*1  # doesn't has concat mask

    d_ob= 4 # the dim of each node features
    d_model = 36 *d_ob #  64  # 256
    nhid = 2 * d_model

    # nhid = 256
    # nhid = 512  # seems to work better than 2*d_model=256
    # nhid = 1024
    nlayers = 2

    # nhead = 16 # seems to work better
    nhead = 4  # 8, 16, 32

    # nhead = 1  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5

    dropout = 0.3 #0.5 (81.2) # 0.3 (81.5)

    if dataset == 'P12':
        max_len = 215
    elif dataset == 'P19':
        max_len = 60
    elif dataset == 'eICU':
        max_len = 300

    aggreg = 'mean'
    # aggreg = 'max'

    n_classes = 2
    # MAX = d_model
    MAX = 100

    n_runs = 1 # change this from 1 to 1, in order to save debugging time.
    n_splits = 5 # change this from 5 to 1, in order to save debugging time.
    subset = False  # use subset for better debugging in local PC, which only contains 1200 patients

    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    for k in range(n_splits):
        # k = 1
        split_idx = k+1
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

        # prepare the data:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
                                                                  baseline=baseline, dataset=dataset)
        # Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path) # use fixed split
        # Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path=None) # use random split
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))


        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        print(T, F, D)

        # get mean, std stats from train set
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))  # shape: (9600, 215, 36)
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))  # shape: (9600, 9)

        # feed features to tensor. This step can be improved
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        """Z-score Normalization. Before this step, we can remove Direct current shift (minus the average) """
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                     stdf, ms, ss)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                          ss)
        """After normalization, a large proportion (more than half) of the values are becoming 1."""
        print(Ptrain_tensor.shape, Ptrain_static_tensor.shape, Ptrain_time_tensor.shape, ytrain_tensor.shape)
        # the shapes are: torch.Size([960, 215, 72]) torch.Size([960, 9]) torch.Size([960, 215, 1]) torch.Size([960])

        """calculate/load global structure """
        print('try to load global structure, if do not exist, calculate it and save.')
        # try:
        #     global_structure = np.load(base_path + '/splits/global_strucutre' + str(split_idx) +'_normalize.npy')
        #     global_structure = torch.from_numpy(global_structure)
        #     print('load global structure')
        # except:

        global_structure = generate_global_structure_diffuse(Ptrain_tensor, K=10) #generate_global_structure
            #  # softmax
            # # global_structure = torch.nn.functional.softmax(global_structure, dim=0)
            # np.save(base_path + '/splits/global_strucutre' + str(split_idx) +'_normalize.npy', global_structure)
            # print('generate global structure for split {}, saved'.format(split_idx))

        # remove part of variables in validation and test set
        if missing_ratio > 0:
            num_all_features = Pval_tensor.shape[
                2]  # int(Pval_tensor.shape[2] / 2)#  Pval_tensor.shape[2] #   # divided by 2, because of mask
            num_missing_features = round(missing_ratio * num_all_features)
            if feature_removal_level == 'sample':
                for i, patient in enumerate(Pval_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # values
                    # patient[:, idx + num_all_features] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # masks
                    Pval_tensor[i] = patient
                for i, patient in enumerate(Ptest_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)  # values
                    # patient[:, idx + num_all_features] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)  # masks
                    Ptest_tensor[i] = patient
            elif feature_removal_level == 'set':
                if dataset == 'P12':
                    dataset_prefix = ''
                elif dataset == 'P19':
                    dataset_prefix = 'P19_'
                elif dataset == 'eICU':
                    dataset_prefix = 'eICU_'
                density_score_indices = np.load('baselines/saved/' + dataset_prefix + 'density_scores.npy',
                                                allow_pickle=True)[:, 0]
                # num_missing_features = num_missing_features * 2
                idx = density_score_indices[:num_missing_features].astype(int)
                Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1],
                                                     num_missing_features)  # values
                # Pval_tensor[:, :, idx + num_all_features] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)  # masks
                Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1],
                                                      num_missing_features)  # values
                # Ptest_tensor[:, :, idx + num_all_features] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)  # masks
        # convert to (seq_len, batch, feats)
        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)  # shape: [215, 960, 72]
        Pval_tensor = Pval_tensor.permute(1, 0, 2)
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

        """Mask out some variables: [ 8, 14, 17, 20, 21, 22, 29, 30, 33] (sparse ratio>0.1)
        1. We train on all sensors, but testing with partial sensors
        2. Train and test on partial sensors. """
        # variable = [ 8, 14, 17, 20, 21, 22, 29, 30, 33]  # AUROC =76.5 if only keep these variables; AUROC = 83.4 if remove these variables
        # variable = [ 8,  9, 10, 13, 14, 17, 18, 20, 21, 22, 24, 25, 27, 29, 30, 33, 35] #AUROC =83.9 if only keep these variables; AUROC =75.1 if remove these variables
        # variable = [9, 10, 13, 18, 24, 25, 27, 35] # AUROC =79.7 if only keep these variables;AUROC = 78.3 if remove these variables


        # variable = [ 8, 10, 14, 17, 20, 21, 22, 27, 29, 30, 33] # ratio>0.047 (appearted 10 observations)
        # for i in range(36):
        #     # if i not in variable:
        #     if i in variable:
        #     #     #Ptrain_tensor[:, :, i]=0
        #         Pval_tensor[:, :, i] = 0
        #         Ptest_tensor[:, :, i] = 0

        # convert to (seq_len, batch)
        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)  # shape: [215, 9590]
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)


        for m in range(n_runs):
            print('- - Run %d - -' % (m + 1))
            """"Xiang: until here, all the above processing are the same as TX_irregular_splits_subset.py"""""

            # instantiate model
            # model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
            #                           d_static, MAX, 0.5, aggreg, n_classes)
            # HGT_latconcat, Raindrop,
            """d_inp = 36 * 1 ;        d_model = 36 * 2;        nhid = 2 * d_model"""
            model = Raindrop_v2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                      d_static, MAX, 0.5, aggreg, n_classes, global_structure)

            # model = Simple_classifier(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
            #                           d_static, MAX, 0.5, aggreg, n_classes, global_structure)

            model = model.cuda()

            criterion = torch.nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            """Keep scheduler turned on"""
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                   patience=1, threshold=0.0001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]

            strategy = 2 # 2 # strategy 2 indeed works better!
            balance = True # set as True works better.

            """Upsampling, increase the number of positive samples"""
            # Strategy 2: permute randomly each index set at each epoch, and expand x3 minority set
            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            batch_size = 128  # balanced batch size
            if strategy == 1:
                n_batches = 30  # number of batches to process per epoch
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])

            best_aupr_val, best_auc_val, best_acc_val =0.0, 0.0, 0.0
            print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))

            #         optimizer = NoamOpt(d_model, 5.0, 500, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

            start = time.time()
            if wandb:
                wandb.watch(model)
            for epoch in range(num_epochs):
                if epoch==0: # the first batch don't propogate error
                    model.eval()
                else:
                    model.train()

                if strategy == 2:
                    """shuffle the index of positive and negative samples"""
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                """In each epoch, first shuffle the samples, then take the first n_batches*int(batch_size / 2) for training"""
                for n in range(n_batches):
                    if strategy == 1:
                        if balance==True:
                            idx = random_sample(idx_0, idx_1, batch_size)  # balance
                        elif balance==False:
                            all_ids = np.concatenate([idx_0, idx_1], axis=0)  # imbalance
                            idx = np.random.choice(all_ids, size=batch_size, replace=False)

                    elif strategy == 2:
                        """In each batch=128, 64 positive samples, 64 negative samples"""
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)

                    P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                           Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()

                    """Shape [128]. Length means the number of timepoints in each sample, for all samples in this batch"""
                    lengths = torch.sum(Ptime > 0, dim=0)

                    """Use two different ways to check the results' consistency"""


                    outputs, local_structure_regularization, _ = model.forward(P, Pstatic, Ptime, lengths)
                    # outputs = evaluate_standard(model, P, Ptime, Pstatic)
                    """Input to the model: 
                    P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
                    Pstatic: [128, 9]: this one doesn't matter; static features
                    Ptime: [215, 128]: the timestamps
                    lengths: [128]: the number of nonzero recordings.
                    """
                    optimizer.zero_grad()
                    loss = criterion(outputs, y) + 0.2*local_structure_regularization
                    loss.backward()
                    optimizer.step()
                """Training performance"""
                train_probs = torch.squeeze(nn.functional.softmax(outputs,dim=1))

                train_probs = train_probs.cpu().detach().numpy()
                # train_probs = softmax(outputs, 1)
                train_y = y.cpu().detach().numpy()
                train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                train_auprc = average_precision_score(train_y, train_probs[:, 1])

                # scheduler.step(train_auroc)  # this makes it worse

                # print("Train: Epoch %d, train loss:%.4f, train_auprc: %.2f, train_auroc: %.2f" % (
                # epoch, loss.item(),  train_auprc * 100, train_auroc * 100))

                if wandb:
                    wandb.log({"train_loss": loss.item(),  "train_auprc": train_auprc, "train_auroc": train_auroc}) #"epoch":epoch,
                if epoch == num_epochs-1:
                    print("training CM", confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=[0, 1]))

                    # train_auc_val = roc_auc_score(y, probs[:, 1])
                    # train_aupr_val = average_precision_score(y, probs[:, 1])

                """Validation"""
                model.eval()
                if epoch % 1 == 0:
                    with torch.no_grad():
                        out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor)
                        # out_val = torch.squeeze(torch.sigmoid(out_val))

                        """the torch.nn.CrossEntropyLoss() already contains softmax, so we don't need softmax here"""
                        out_val = out_val.detach().cpu().numpy()


                        # denoms = np.sum(np.exp(out_val), axis=1).reshape((-1, 1))
                        # probs = np.exp(out_val) / denoms
                        # ypred = np.argmax(out_val, axis=1)

                        val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)))

                        """here use softmax for the calculation of metrics"""
                        out_val = softmax(out_val, axis=1)
                            # torch.squeeze(nn.functional.softmax(out_val, dim=1))

                        auc_val = roc_auc_score(yval, out_val[:, 1])
                        aupr_val = average_precision_score(yval, out_val[:, 1])
                        acc_val = accuracy_score(yval, np.argmax(out_val, axis=1))
                        print("Validataion: Epoch %d,  val_loss:%.4f, acc_val:%.2f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                          val_loss.item(), acc_val*100, aupr_val * 100, auc_val * 100))

                        if wandb:
                            wandb.log({ "val_loss": val_loss.item(), "val_auprc": aupr_val,
                                       "val_auroc": auc_val})

                        if epoch == num_epochs - 1:
                            print("Val CM", confusion_matrix(yval, np.argmax(out_val, axis=1), labels=[0, 1]))


                        scheduler.step(auc_val)
                        # save model
                        # if aupr_val > best_aupr_val:
                        #     best_aupr_val = aupr_val
                        if auc_val > best_auc_val:
                            best_auc_val = auc_val
                            print(
                                "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (epoch, aupr_val * 100, auc_val * 100))
                            torch.save(model.state_dict(), model_path + arch + '_' + str(split_idx) + '.pt')

                # if epoch == 3:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print('-- Estimated train time: %.3f mins --' % (time_elapsed / 60.0 / 4 * num_epochs))

            end = time.time()
            time_elapsed = end - start
            print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

            """testing"""
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
                print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
                print('classification report', classification_report(ytest, ypred))
                print(confusion_matrix(ytest, ypred, labels=[0, 1]))

                if wandb:
                    wandb.log({ "z_test_auroc": auc, "z_test_auprc": aupr})
            # store
            acc_arr[k, m] = acc * 100
            auprc_arr[k, m] = aupr * 100
            auroc_arr[k, m] = auc * 100

    # pick best performer for each split based on max AUPRC
    idx_max = np.argmax(auprc_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]

    print("missing ratio:{}, split type:{}, reverse:{}, using baseline:{}".format(missing_ratio, split, reverse, baseline))


    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    print('------------------------------------------')
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))
    print('\nAbove results for missing ratio: %d\n\n\n' % missing_ratio)

    # Mark the run as finished
    if wandb:
        wandb.finish()

    # # save in numpy file
    # np.save('./results/' + arch + '_phy12.npy', [acc_vec, auprc_vec, auroc_vec])


