"""
Code originates from original mTAND code on GitHub repository https://github.com/reml-lab/mTAN.

Original paper:
Shukla, Satya Narayan, and Benjamin Marlin. "Multi-Time Attention Networks for Irregularly Sampled Time Series."
International Conference on Learning Representations. 2020.
"""

#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from random import SystemRandom
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_enc')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=128) # 128
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', default=True, action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--num-heads', type=int, default=1)
parser.add_argument('--freq', type=float, default=10.)

parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0')
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
args = parser.parse_args(args=[])

if __name__ == '__main__':
    """"0 means no missing (full observations); 1.0 means no observation, all missed"""
    if args.withmissingratio:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    for missing_ratio in missing_ratios:
        acc_all = []
        auc_all = []
        aupr_all = []
        precision_all = []
        recall_all = []
        F1_all = []
        upsampling_batch = True

        split_type = args.splittype  # possible values: 'random', 'age', 'gender' ('age' not possible for dataset 'eICU')
        reverse_ = args.reverse  # False, True
        feature_removal_level = args.feature_removal_level  # 'sample', 'set'
        num_runs = 5
        for r in range(num_runs):
            experiment_id = int(SystemRandom().random() * 100000)
            if r == 0:
                print(args, experiment_id)
            seed = args.seed
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            print('we are using: {}'.format(device))

            args.classif = True
            args.niters = 20  # number of epochs
            dataset = args.dataset  # possible values: 'P12', 'P19', 'eICU', 'PAM'
            print('Dataset used: ', dataset)

            data_obj = utils.get_data(args, dataset, device, args.quantization, upsampling_batch, split_type,
                                      feature_removal_level, missing_ratio, reverse=reverse_, predictive_label=args.predictive_label)

            train_loader = data_obj["train_dataloader"]
            test_loader = data_obj["test_dataloader"]
            val_loader = data_obj["val_dataloader"]
            dim = data_obj["input_dim"]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
                n_classes = 2
            elif dataset == 'PAM':
                n_classes = 8

            # model
            if args.enc == 'mtan_enc':
                rec = models.enc_mtan_classif(
                    dim, torch.linspace(0, 1., 128), args.rec_hidden, args.embed_time, args.num_heads,
                    args.learn_emb, args.freq, device=device, n_classes=n_classes).to(device)

            elif args.enc == 'mtan_enc_activity':
                rec = models.enc_mtan_classif_activity(
                    dim, args.rec_hidden, args.embed_time,
                    args.num_heads, args.learn_emb, args.freq, device=device).to(device)

            params = (list(rec.parameters()))
            if r == 0:
                print('parameters:', utils.count_parameters(rec))
            optimizer = optim.Adam(params, lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            if args.fname is not None:
                checkpoint = torch.load(args.fname)
                rec.load_state_dict(checkpoint['rec_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('loading saved weights', checkpoint['epoch'])

            best_val_loss = float('inf')
            total_time = 0.
            best_aupr_val = 0
            saved_model_path = 'best_model_val_aupr.pt'
            print('\n------------------\nRUN %d: Training started\n------------------' % r)
            for itr in range(1, args.niters + 1):
                train_loss = 0
                train_n = 0
                train_acc = 0
                start_time = time.time()
                for train_batch, label in train_loader:
                    train_batch, label = train_batch.to(device), label.to(device)
                    batch_len = train_batch.shape[0]
                    observed_data, observed_mask, observed_tp \
                        = train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]
                    out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
                    if args.classify_pertp:
                        N = label.size(-1)
                        out = out.view(-1, N)
                        label = label.view(-1, N)
                        _, label = label.max(-1)
                        loss = criterion(out, label.long())
                    else:
                        loss = criterion(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_len
                    train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
                    train_n += batch_len

                total_time += time.time() - start_time

                # validation set
                val_loss, val_acc, val_auc, val_aupr, val_precision, val_recall, val_F1 = \
                    utils.evaluate_classifier(rec, val_loader, args=args, dim=dim, dataset=dataset)
                best_val_loss = min(best_val_loss, val_loss)

                print(
                    'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, '
                    'val_AUPRC: {:.2f}, val_precision: {:.2f},val_recall: {:.2f},val_F1: {:.2f},'
                    .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                            val_aupr * 100, val_precision * 100, val_recall * 100, val_F1 * 100))

                # save the best model based on 'aupr'
                if val_aupr > best_aupr_val:
                    best_aupr_val = val_aupr
                    torch.save(rec, saved_model_path)

            print('\n------------------\nRUN %d: Training finished\n------------------' % r)

            # test set
            rec = torch.load(saved_model_path)
            test_loss, test_acc, test_auc, test_aupr, test_precision, test_recall, test_F1 = \
                utils.evaluate_classifier(rec, test_loader, args=args, dim=dim, dataset=dataset)
            print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f, auc_precision: %.2f, auc_recall: %.2f, auc_F1: %.2f\n" % (
            test_acc * 100, test_aupr * 100, test_auc * 100, test_precision * 100, test_recall * 100, test_F1 * 100))

            acc_all.append(test_acc * 100)
            auc_all.append(test_auc * 100)
            aupr_all.append(test_aupr * 100)
            if dataset == 'PAM':
                precision_all.append(test_precision * 100)
                recall_all.append(test_recall * 100)
                F1_all.append(test_F1 * 100)

        # print mean and std of all metrics
        acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
        mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
        mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
        mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
        print('------------------------------------------')
        print("split:{}, set/sample-level: {}, missing ratio:{}".format(split_type, feature_removal_level, missing_ratio))
        print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
              args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)
        print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
        print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
        print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))
        if dataset == 'PAM':
            precision_all, recall_all, F1_all = np.array(precision_all), np.array(recall_all), np.array(F1_all)
            mean_precision, std_precision = np.mean(precision_all), np.std(precision_all)
            mean_recall, std_recall = np.mean(recall_all), np.std(recall_all)
            mean_F1, std_F1 = np.mean(F1_all), np.std(F1_all)
            print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
            print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
            print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))
