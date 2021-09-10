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
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_enc')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', default=True, action='store_true',
                    help="Include binary classification loss")

parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--num-heads', type=int, default=1)
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--classify-pertp', action='store_true')
# args = parser.parse_args()
args = parser.parse_args(args=[])

if __name__ == '__main__':
    """"0 means no missing (full observations); 1.0 means no observation, all missed"""
    #     missing_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    missing_ratios = [0.2]
    for missing_ratio in missing_ratios:
        acc_all = []
        auc_all = []
        aupr_all = []
        upsampling_batch = True

        split_type = 'gender'  # possible values: 'random', 'age', 'gender' ('age' not possible for dataset 'eICU')
        reverse_ = False  # False, True
        feature_removal_level = 'set'  # possible values: 'sample', 'set'
        num_runs = 5
        for r in range(num_runs):
            experiment_id = int(SystemRandom().random() * 100000)
            if r == 0:
                print(args, experiment_id)
            seed = args.seed
            # torch.manual_seed(seed)
            # np.random.seed(seed)
            # torch.cuda.manual_seed(seed)
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            print('we are using:{}'.format(device))

            # device = 'cpu'  # todo
            args.classif = True
            args.niters = 20  # number of epochs
            dataset = 'eICU'     # possible values: 'P12', 'P19', 'eICU'

            data_obj = utils.get_data(args, dataset, device, args.quantization, upsampling_batch, split_type,
                                      feature_removal_level, missing_ratio, reverse=reverse_)
            # elif args.dataset == 'mimiciii':
            #     data_obj = utils.get_mimiciii_data(args)
            # elif args.dataset == 'activity':
            #     data_obj = utils.get_activity_data(args, 'cpu')

            train_loader = data_obj["train_dataloader"]
            test_loader = data_obj["test_dataloader"]
            val_loader = data_obj["val_dataloader"]
            dim = data_obj["input_dim"]

            # model
            if args.enc == 'mtan_enc':
                rec = models.enc_mtan_classif(
                    dim, torch.linspace(0, 1., 128), args.rec_hidden,
                    args.embed_time, args.num_heads, args.learn_emb, args.freq, device=device).to(device)

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
                # avg_reconst, avg_kl, mse = 0, 0, 0
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
                val_loss, val_acc, val_auc, val_aupr = utils.evaluate_classifier(rec, val_loader, args=args, dim=dim)
                best_val_loss = min(best_val_loss, val_loss)

                print(
                    'VALIDATION: Iter: {}, loss: {:.4f}, acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.2f}, val_AUROC: {:.2f}, val_AUPRC: {:.2f}'
                    .format(itr, train_loss / train_n, train_acc / train_n, val_loss, val_acc * 100, val_auc * 100,
                            val_aupr * 100))

                # save the best model based on 'aupr'
                if val_aupr > best_aupr_val:
                    best_aupr_val = val_aupr
                    torch.save(rec, saved_model_path)

                # if itr % 100 == 0 and args.save:
                #     torch.save({
                #         'args': args,
                #         'epoch': itr,
                #         'rec_state_dict': rec.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         # 'loss': -loss,
                #     }, args.dataset + '_' +
                #         args.enc + '_' +
                #         #args.dec + '_' +
                #         str(experiment_id) +
                #         '.h5')
            print('\n------------------\nRUN %d: Training finished\n------------------' % r)

            # test set
            rec = torch.load(saved_model_path)
            test_loss, test_acc, test_auc, test_aupr = utils.evaluate_classifier(rec, test_loader, args=args, dim=dim)
            print("TEST: test_acc: %.2f, aupr_test: %.2f, auc_test: %.2f\n" % (
            test_acc * 100, test_aupr * 100, test_auc * 100))

            acc_all.append(test_acc * 100)
            auc_all.append(test_auc * 100)
            aupr_all.append(test_aupr * 100)

            # print(best_val_loss)
            # print(total_time)

        # print mean and std of all metrics
        acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
        mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
        mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
        mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
        print('------------------------------------------')
        print(
            "split:{}, set/sample-level: {}, missing ratio:{}".format(split_type, feature_removal_level, missing_ratio))
        print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
        print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
        print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))