# UniMiB dataset: http://www.sal.disco.unimib.it/technologies/unimib-shar/


import numpy as np
import scipy.io as sc
from sklearn import preprocessing
wandb = False #False

if wandb:
    import wandb
    wandb.login(key=str('14734fe9c5574e019e8f517149a20d6fe1b2fd0d'))
    config = wandb.config
    # run = wandb.init(project='WBtest', config={'wandb_nb':'wandb_three_in_one_hm'})
    run = wandb.init(project='Raindrop', entity='XZ', config={'wandb_nb':'wandb_three_in_one_hm'})
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from models import  TransformerModel2, Simple_classifier
from utils_baselines import *
from scipy.sparse import random


# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def extract(input, n_classes, n_fea, time_window, moving):
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data


def split_id(n = 12000):
    # n = 11988  # original 12000 patients, remove 12 outliers
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    n_test = n - (n_train + n_val)
    print(n_train, n_val, n_test)
    # Nsplits = 5
    p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    return idx_train, idx_val, idx_test


if __name__ == '__main__':
    arch = 'PAMAP2'
    model_path = '../../models/'

    base_path = '../../PAMAP2data/'

    """"PAMAP2 120,000 samples for 6 or 8 subjects. 20,000 samples for each sub. 51 features, 8 activities(0-7)"""
    """100 Hz"""
    # data = sc.loadmat(base_path+"processed_data/AR_8p_8c.mat")
    # dataset = data['AR_8p_8c']  # (1200,000, 52)
    # # dataset = dataset[:20000]
    #
    # feature_all = dataset[:, :17]  # only use one IMU.
    #
    # # feature_all=preprocessing.scale(feature_all)  # Normalization
    #
    # label = dataset[:, 51:52]
    # dataset_new = np.hstack((feature_all, label))
    #
    # n_classes = 8  # 0~7 classes
    # no_feature = 17  # the number of the features
    # segment_length = 600  # selected time window; 16=160*0.1
    #
    # data_seg = extract(dataset_new, n_classes=n_classes, n_fea=no_feature, time_window=segment_length, moving=(segment_length/2))  # 50% overlapping
    # print('After segmentation, the shape of the data:', data_seg.shape)


    # Pdict_list = data_seg[:, :segment_length * no_feature]  #.reshape([-1, segment_length, no_feature])  # (4000, 600, 17)
    # arr_outcome = data_seg[:, -1:]
    # # save Pdict_list.py and arr_outcome.npy
    # np.save('../../PAMAP2data/processed_data/PTdict_list.npy', Pdict_list, allow_pickle=True)
    # np.save('../../PAMAP2data/processed_data/arr_outcomes.npy', arr_outcome, allow_pickle=True)
    #
    # """remove 90% values in Pdict_list: sparse ratio 90% """
    # sparse_mask = random(data_seg.shape[0], segment_length * no_feature,
    #                      random_state=1, format='csr', density=0.4)  # density \in [0,1]
    # sparse_mask = sparse_mask.astype(bool)  #.astype(int)
    #
    # # # sparse_mask = sparse_mask.tocsr()
    # # np.save('./saved/PAMAP2_sparse_mask.npy', sparse_mask, allow_pickle=True)
    # # sparse_mask = np.load('./saved/PAMAP2_sparse_mask.npy', allow_pickle=True)
    #
    # sparse_mask = sparse_mask.todense()
    # Pdict_list = np.multiply(Pdict_list, sparse_mask)
    # Pdict_list = np.array(Pdict_list).reshape([-1, segment_length, no_feature])
    # # save final Pdict_list.py
    # np.save('../../PAMAP2data/processed_data/PTdict_list.npy', Pdict_list, allow_pickle=True)

    Pdict_list = np.load('../../PAMAP2data/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcome = np.load('../../PAMAP2data/processed_data/arr_outcomes.npy', allow_pickle=True)

    idx_train, idx_val, idx_test = split_id(5333)   #data_seg.shape[0])
    # np.save('../../PAMAP2data/splits/PAMAP2_split_5.npy', (idx_train, idx_val, idx_test))
    # idx_train, idx_val, idx_test = np.load('../../PAMAP2data/splits/PAMAP2_split_1.npy', allow_pickle=True)

    # feature_all = data_seg[:, :17]  # only use one IMU. (4000, 17)
    # label = data_seg[:, 51:52]

    # #z-score
    # # feature_all=preprocessing.scale(feature_all)
    # no_fea=feature_all.shape[-1]
    # label_all=one_hot(label)

    split = 'random'  # possible values: 'random', 'age', 'gender'
    reverse = False
    baseline = True
    static_info = None  # if dataset is PAMAP2, set this as None

    feature_removal_level = 'set'   # possible values: 'sample', 'set'
    # missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    missing_ratios = [0.1]
    for missing_ratio in missing_ratios:
        # training/model params
        num_epochs = 20
        learning_rate = 0.001

        d_static = 9

        d_inp = 17  # doesn't has concat mask

        d_model = 36  # 256
        nhid = 2 * d_model
        # nhid = 256
        # nhid = 512  # seems to work better than 2*d_model=256
        # nhid = 1024
        nlayers = 1 #2  # the layer doesn't really matters

        # nhead = 16 # seems to work better
        nhead = 1  # 8, 16, 32

        dropout = 0.3

        max_len = 1000

        aggreg = 'mean'
        # aggreg = 'max'

        n_classes = 8
        # MAX = d_model
        MAX = 100

        n_runs = 5  # change this from 1 to 1, in order to save debugging time.
        n_splits = 1  # change this from 5 to 1, in order to save debugging time.
        subset = False  # use subset for better debugging in local PC, which only contains 1200 patients

        """split = 'random', 'age', 'gender"""
        """reverse= True: male, age<65 for training. 
         reverse=False: female, age>65 for training"""
        """baseline=True: run baselines. False: run our model (Raindrop)"""


        acc_arr = np.zeros((n_splits, n_runs))
        auprc_arr = np.zeros((n_splits, n_runs))
        auroc_arr = np.zeros((n_splits, n_runs))
        for k in range(n_splits):
            split_idx = k + 1
            print('Split id: %d' % split_idx)
            # if subset==True:
            #     split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            # else:
            #     split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
            split_path = None

            # prepare the data:
            # Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
            #                                                           baseline=baseline)


            # y = data_seg[:, -1:]  # (4000, 1)
            y = arr_outcome

            Ptrain = Pdict_list[idx_train]
            Pval = Pdict_list[idx_val]
            Ptest = Pdict_list[idx_test]
            ytrain = y[idx_train]
            yval = y[idx_val]
            ytest = y[idx_test]

            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))


            T, F = Ptrain[0].shape
            D = 1 # len(Ptrain[0]['extended_static'])
            print(T, F, D)

            # get mean, std stats from train set
            Ptrain_tensor = Ptrain  # np.zeros((len(Ptrain), T, F))  # shape: (9600, 215, 36)
            Ptrain_static_tensor =  np.zeros((len(Ptrain), D))  #np.zeros((len(Ptrain), D))  # shape: (9600, 9)

            # feed features to tensor. This step can be improved
            # for i in range(len(Ptrain)):
            #     Ptrain_tensor[i] = Ptrain[i]['arr']
            #     Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            """Z-score Normalization. Before this step, we can remove Direct current shift (minus the average) """
            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor)

            Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)
            """After normalization, a large proportion (more than half) of the values are becoming 1."""
            print(Ptrain_tensor.shape, Ptrain_static_tensor, Ptrain_time_tensor.shape, ytrain_tensor.shape)
            # the shapes are: torch.Size([960, 215, 72]) torch.Size([960, 9]) torch.Size([960, 215, 1]) torch.Size([960])

            # remove part of variables in validation and test set
            if missing_ratio > 0:
                num_all_features = int(Pval_tensor.shape[2] / 2)# Pval_tensor.shape[2] # # divided by 2, because of mask
                num_missing_features = round(missing_ratio * num_all_features)
                if feature_removal_level == 'sample':
                    for i, patient in enumerate(Pval_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        # idx = np.array(list(range(num_missing_features)))
                        patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # values
                        # patient[:, idx + num_all_features] = torch.zeros(Pval_tensor.shape[1], num_missing_features)  # masks
                        Pval_tensor[i] = patient
                    for i, patient in enumerate(Ptest_tensor):
                        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                        patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)   # values
                        # patient[:, idx + num_all_features] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)  # masks
                        Ptest_tensor[i] = patient
                elif feature_removal_level == 'set':
                    # density_score_indices = np.load('saved/density_scores.npy', allow_pickle=True)[:, 0]
                    density_score_indices = np.load('saved/IG_density_scores_' + arch + '.npy', allow_pickle=True)[:, 0]

                    # num_missing_features = num_missing_features * 2
                    idx = density_score_indices[:num_missing_features].astype(int)
                    Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)   # values
                    # Pval_tensor[:, :, idx + num_all_features] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)  # masks
                    Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)   # values
                    # Ptest_tensor[:, :, idx + num_all_features] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)  # masks

            # convert to (seq_len, batch, feats)
            Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)  # shape: [215, 960, 72]
            Pval_tensor = Pval_tensor.permute(1, 0, 2)
            Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

            # convert to (seq_len, batch)
            Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
            Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
            Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

            for m in range(n_runs):
                print('- - Run %d - -' % (m + 1))
                model = Simple_classifier(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                                          d_static, MAX, 0.5, aggreg, n_classes, static=False)

                # model = TransformerModel2(d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
                #                           d_static, MAX, 0.5, aggreg, n_classes, static=False)

                model = model.cuda()

                criterion = torch.nn.CrossEntropyLoss().cuda()
                # loss_func = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                                       patience=1, threshold=0.0001, threshold_mode='rel',
                                                                       cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)


                idx_0 = np.where(ytrain == 0)[0]
                idx_1 = np.where(ytrain == 1)[0]

                strategy = 3  # 2

                """Upsampling, increase the number of positive samples"""
                # Strategy 2: permute randomly each index set at each epoch, and expand x3 minority set
                n0, n1 = len(idx_0), len(idx_1)
                expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
                expanded_n1 = len(expanded_idx_1)

                batch_size = 128*4  # balanced batch size
                if strategy == 1 or strategy ==3 :
                    n_batches = 10  # number of batches to process per epoch
                elif strategy == 2:
                    K0 = n0 // int(batch_size / 2)
                    K1 = expanded_n1 // int(batch_size / 2)
                    n_batches = np.min([K0, K1])

                best_aupr_val = best_auc_val = 0.0
                print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))
                #         optimizer = NoamOpt(d_model, 5.0, 500, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

                start = time.time()
                if wandb:
                    wandb.watch(model)
                for epoch in range(num_epochs):
                    model.train()

                    if strategy == 2:
                        """shuffle the index of positive and negative samples"""
                        np.random.shuffle(expanded_idx_1)
                        I1 = expanded_idx_1
                        np.random.shuffle(idx_0)
                        I0 = idx_0
                        # # random shuffling of expanded_idx_1, idx_0
                        # ep1 = np.random.permutation(expanded_n1)
                        # p0 = np.random.permutation(n0)
                        # I1 = expanded_idx_1[ep1]
                        # I0 = idx_0[p0]
                    """In each epoch, first shuffle the samples, then take the first n_batches*int(batch_size / 2) for training"""


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


                        P, Ptime, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                                               None, ytrain_tensor[idx].cuda()

                        """Shape [128]. Length means the number of timepoints in each sample, for all samples in this batch"""
                        lengths = torch.sum(Ptime > 0, dim=0)

                        """Use two different ways to check the results' consistency"""
                        # outputs = model.forward(P, Pstatic, Ptime, lengths)
                        outputs = evaluate_standard(model, P, Ptime, Pstatic, n_classes=n_classes, static=static_info)

                        # if epoch == 0:
                        #     optimizer.zero_grad()
                        #     loss = criterion(outputs, y)
                        # elif epoch>0:  # Don't train the model at epoch==0

                        optimizer.zero_grad()
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                    """Training performance"""
                    # train_probs = torch.squeeze(torch.sigmoid(outputs))
                    train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))

                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()
                    # train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                    # train_auprc = average_precision_score(train_y, train_probs[:, 1])
                    train_auroc = roc_auc_score(one_hot(train_y), train_probs)
                    train_auprc = average_precision_score(one_hot(train_y), train_probs)


                    # print("Train: Epoch %d, train loss:%.4f, train_auprc: %.2f, train_auroc: %.2f" % (
                    # epoch, loss.item(),  train_auprc * 100, train_auroc * 100))
                    if wandb:
                        wandb.log({ "train_loss": loss.item(), "train_auprc": train_auprc, "train_auroc": train_auroc})
                    # if epoch == 0 or epoch == num_epochs-1:
                    #     print(confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=list(range(n_classes))))

                    """Use the last """
                    """Validation"""
                    model.eval()
                    if epoch ==0 or epoch % 1 == 0:
                        with torch.no_grad():
                            out_val = evaluate_standard(model, Pval_tensor, Pval_time_tensor, Pval_static_tensor,
                                                        n_classes=n_classes, static=static_info)
                            # out_val = torch.squeeze(torch.sigmoid(out_val))
                            # denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
                            # probs = np.exp(out_test) / denoms

                            out_val = out_val.detach().cpu().numpy()

                            val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                            # auc_val = roc_auc_score(yval, out_val[:, 1])
                            # aupr_val = average_precision_score(yval, out_val[:, 1])
                            auc_val = roc_auc_score(one_hot(yval), out_val)
                            aupr_val = average_precision_score(one_hot(yval), out_val)
                            print("Validataion: Epoch %d,  val_loss:%.4f, aupr_val: %.2f, auc_val: %.2f" % (epoch,
                              val_loss.item(), aupr_val * 100, auc_val * 100))
                            # print(confusion_matrix(yval, np.argmax(out_val, axis=1),))

                            if wandb:
                                wandb.log({ "val_loss": val_loss.item(), "val_auprc": aupr_val, "val_auroc": auc_val})

                            scheduler.step(aupr_val)
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
                    out_test = evaluate(model, Ptest_tensor, Ptest_time_tensor, Ptest_static_tensor,
                                        n_classes=n_classes, static=static_info).numpy()
                    ypred = np.argmax(out_test, axis=1)


                    denoms = np.sum(np.exp(out_test), axis=1).reshape((-1, 1))
                    probs = np.exp(out_test) / denoms

                    # auc = roc_auc_score(ytest, probs[:, 1])
                    # aupr = average_precision_score(ytest, probs[:, 1])
                    auc = roc_auc_score(one_hot(ytest), probs)
                    aupr = average_precision_score(one_hot(ytest), probs)
                    acc = np.sum(ytest.ravel() == ypred.ravel()) / ytest.shape[0]
                    print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
                    print('classification report', classification_report(ytest, ypred))
                    print(confusion_matrix(ytest, ypred, labels=list(range(n_classes))))

                    # np.save('../results/MLP_test_probs.npy', probs, allow_pickle=True)
                    # np.save('../results/MLP_test_ytruth.npy', ytest, allow_pickle=True)
                    # print('data saved')
                    # break

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

        # Mark the run as finished
        if wandb:
            wandb.finish()


    print('good work')

