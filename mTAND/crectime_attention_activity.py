#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from random import SystemRandom
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='activity')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'activity':
        data_obj = utils.get_activity_data(args, 'cpu')
   
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., 50), args.latent_dim, args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)
        
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., 50), args.latent_dim, args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)

    classifier = nn.Sequential(
            nn.Linear(args.latent_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 11))
    classifier = classifier.to(device)
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = train_batch.shape[0]
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            print(z0.shape)
            pred_y = classifier(z0)
            print(pred_y.shape)
            # compute loss
            if args.classify_pertp:
                N = label.size(-1)
                out = pred_y.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                ce_loss = criterion(out, label.long()) 
            else:
                loss = criterion(out, label) 
                
            loss = ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_ce_loss += ce_loss.item() * batch_len
            train_acc += torch.mean((out.argmax(1) == label).float()).item() * batch_len
            train_n += batch_len
        total_time += time.time() - start_time
        val_loss, val_acc, val_auc = utils.evaluate_classifier(
            rec, val_loader, dec=dec, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            rec, test_loader, dec=dec, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        print('Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(itr, train_recon_loss/train_n, train_ce_loss/train_n, 
                      train_acc/train_n, mse/train_n, val_loss, val_acc, test_acc, test_auc))
        
            
        if itr % 100 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'classifier_state_dict': classifier_state_dict,
                'loss': -loss,
            }, args.dataset + '_' + 
                args.enc + '_' + 
                args.dec + '_' + 
                str(experiment_id) +
                '.h5')
    print(best_val_loss)
    print(total_time)
