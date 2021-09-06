# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, confusion_matrix

import numpy as np
from physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2
from sklearn import model_selection
from sklearn import metrics
from person_activity import PersonActivity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cuda"):
    mse, test_n = 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):   # todo
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)

    acc = np.mean(pred.argmax(1) == true)
    auc = metrics.roc_auc_score(true, pred[:, 1]) if not args.classify_pertp else 0.
    aupr = average_precision_score(true, pred[:, 1]) if not args.classify_pertp else 0.

    # print('Non-zero predictions = ', np.count_nonzero(np.argmax(pred, axis=1)))
    # print(confusion_matrix(true, np.argmax(pred, axis=1), labels=[0, 1]))

    return test_loss/pred.shape[0], acc, auc, aupr


def get_mimiciii_data(args):
    input_dim = 12
    x = np.load('../../../neuraltimeseries/Dataset/final_input3.npy')
    y = np.load('../../../neuraltimeseries/Dataset/final_output3.npy')
    x = x[:, :25]
    x = np.transpose(x, (0, 2, 1))

    # normalize values and time
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                  :input_dim], x[:, :, input_dim:2*input_dim], x[:, :, -1]
    if np.max(observed_tp) != 0.:
        observed_tp = observed_tp / np.max(observed_tp)

    if not args.nonormalize:
        for k in range(input_dim):
            data_min, data_max = float('inf'), 0.
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            #print(data_min, data_max)
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = (
                observed_vals[:, :, k] - data_min)/data_max
    # set masked out elements back to zero
    observed_vals[observed_mask == 0] = 0
    print(observed_vals[0], observed_tp[0])
    print(x.shape, y.shape)
    kfold = model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0)
    splits = [(train_inds, test_inds)
              for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]
    x_train, y_train = x[splits[args.split][0]], y[splits[args.split][0]]
    test_data_x, test_data_y = x[splits[args.split]
                                 [1]], y[splits[args.split][1]]
    if not args.old_split:
        train_data_x, val_data_x, train_data_y, val_data_y = \
            model_selection.train_test_split(
                x_train, y_train, stratify=y_train, test_size=0.2, random_state=0)
    else:
        frac = int(0.8*x_train.shape[0])
        train_data_x, val_data_x = x_train[:frac], x_train[frac:]
        train_data_y, val_data_y = y_train[:frac], y_train[frac:]

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))
    train_data_combined = TensorDataset(torch.from_numpy(train_data_x).float(),
                                        torch.from_numpy(train_data_y).long().squeeze())
    val_data_combined = TensorDataset(torch.from_numpy(val_data_x).float(),
                                      torch.from_numpy(val_data_y).long().squeeze())
    test_data_combined = TensorDataset(torch.from_numpy(test_data_x).float(),
                                       torch.from_numpy(test_data_y).long().squeeze())
    train_dataloader = DataLoader(
        train_data_combined, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=args.batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}
    return data_objects


def random_sample(idx_0, idx_1, batch_size):
    """
    Returns a balanced sample by randomly sampling without replacement.

    :param idx_0: indices of negative samples
    :param idx_1: indices of positive samples
    :param batch_size: batch size
    :return: indices of balanced batch of negative and positive samples
    """
    """  """
    idx0_batch = np.random.choice(idx_0, size=int(batch_size / 2), replace=False)
    idx1_batch = np.random.choice(idx_1, size=int(batch_size / 2), replace=False)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx


def get_physionet_data(args, device, q, upsampling_batch, split_type, feature_removal_level, missing_ratio, flag=1):
    train_dataset_obj_1 = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=12000,
                                  device=device, set_letter='a')

    train_dataset_obj_2 = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=12000,
                                  device=device, set_letter='b')

    train_dataset_obj_3 = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=12000,
                                  device=device, set_letter='c')

    # # Use custom collate_fn to combine samples with arbitrary time observations.
    # # Returns the dataset along with mask and time steps
    # test_dataset_obj = PhysioNet('data/physionet', train=False,
    #                              quantization=q,
    #                              download=True, n_samples=min(10000, args.n),
    #                              device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    # total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    dataset_1 = train_dataset_obj_1[:len(train_dataset_obj_1)]
    dataset_2 = train_dataset_obj_2[:len(train_dataset_obj_2)]
    dataset_3 = train_dataset_obj_3[:len(train_dataset_obj_3)]

    total_dataset = dataset_1 + dataset_2 + dataset_3

    # if not args.classif:
    #     # Concatenate samples from original Train and Test sets
    #     # Only 'training' physionet samples are have labels.
    #     # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
    #     total_dataset = total_dataset + \
    #         test_dataset_obj[:len(test_dataset_obj)]

    print('len(total_dataset):', len(total_dataset))

    '''
    # calculate and save statistics
    idx_under_65 = []
    idx_over_65 = []
    idx_male = []
    idx_female = []

    P_list = np.load('P_list.npy', allow_pickle=True)

    for i in range(len(P_list)):
        if total_dataset[i][0] == P_list[i]['id']:
            age, gender, _, _, _ = P_list[i]['static']
            if age > 0:
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender == 0:
                idx_female.append(i)
            if gender == 1:
                idx_male.append(i)

    np.save('mtand_idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
    np.save('mtand_idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
    np.save('mtand_idx_male.npy', np.array(idx_male), allow_pickle=True)
    np.save('mtand_idx_female.npy', np.array(idx_female), allow_pickle=True)
    '''

    if split_type == 'random':
        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.9,     # 80% train, 10% validation, 10% test
                                                                 shuffle=True)
    elif split_type == 'age' or split_type == 'gender':
        if split_type == 'age':
            idx_train = np.load('mtand_idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load('mtand_idx_over_65.npy', allow_pickle=True)
        else:  # split_type == 'gender':
            idx_train = np.load('mtand_idx_male.npy', allow_pickle=True)
            idx_vt = np.load('mtand_idx_female.npy', allow_pickle=True)

        np.random.shuffle(idx_train)
        np.random.shuffle(idx_vt)
        train_data = [total_dataset[i] for i in idx_train]
        test_data = [total_dataset[i] for i in idx_vt]

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset)
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj_1), args.batch_size), args.n)
    if flag:
        if args.classif:
            if split_type == 'random':
                train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8889, shuffle=True)  # 80% train, 10% validation, 10% test
            elif split_type == 'age' or split_type == 'gender':
                val_data, test_data = model_selection.train_test_split(test_data, train_size=0.5, shuffle=False)

            if feature_removal_level == 'sample':
                num_all_features = 36
                num_missing_features = round(missing_ratio * num_all_features)
                for i, tpl in enumerate(val_data):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    _, _, values, _, _ = tpl
                    tpl = list(tpl)
                    values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
                    tpl[2] = values
                    val_data[i] = tuple(tpl)
                for i, tpl in enumerate(test_data):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    _, _, values, _, _ = tpl
                    tpl = list(tpl)
                    values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
                    tpl[2] = values
                    test_data[i] = tuple(tpl)
            elif feature_removal_level == 'set':
                num_all_features = 36
                num_missing_features = round(missing_ratio * num_all_features)
                dict_params = train_dataset_obj_1.params_dict
                density_scores_names = np.load('density_scores.npy', allow_pickle=True)[:, 2]
                idx = [dict_params[name] for name in density_scores_names[:num_missing_features]]
                for i, tpl in enumerate(val_data):
                    _, _, values, _, _ = tpl
                    tpl = list(tpl)
                    values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
                    tpl[2] = values
                    val_data[i] = tuple(tpl)
                for i, tpl in enumerate(test_data):
                    _, _, values, _, _ = tpl
                    tpl = list(tpl)
                    values[:, idx] = torch.zeros(values.shape[0], num_missing_features)
                    tpl[2] = values
                    test_data[i] = tuple(tpl)

            if upsampling_batch:
                train_data_upsamled = []
                true_labels = np.array(list(map(lambda x: float(x[7]), np.array(train_data)[:, 4])))
                idx_0 = np.where(true_labels == 0)[0]
                idx_1 = np.where(true_labels == 1)[0]
                for _ in range(len(true_labels) // batch_size):
                    indices = random_sample(idx_0, idx_1, batch_size)
                    for i in indices:
                        train_data_upsamled.append(train_data[i])
                train_data = train_data_upsamled

            test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            train_data_combined = variable_time_collate_fn(train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            # print(train_data_combined.size(), test_data_combined.size())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="train",
                                                                                         data_min=data_min, data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="test",
                                                                                        data_min=data_min, data_max=data_max))

    attr_names = train_dataset_obj_1.params
    data_objects = {"dataset_obj": train_dataset_obj_1,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional
    if args.classif:
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_dataloader"] = val_dataloader
    return data_objects


def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                if labels is not None:  # todo
                    combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data


def get_activity_data(args, device):
    n_samples = min(10000, args.n)
    dataset_obj = PersonActivity('data/PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    print(dataset_obj)

    train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8,
                                                             random_state=42, shuffle=True)

    # train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
    # test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

    record_id, tt, vals, mask, labels = train_data[0]
    input_dim = vals.size(-1)

    batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
    test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                  activity=True)
    train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                            random_state=11, shuffle=True)
    train_data_combined = variable_time_collate_fn(
        train_data, device, classify=args.classif, activity=True)
    val_data_combined = variable_time_collate_fn(
        val_data, device, classify=args.classif, activity=True)
    print(train_data_combined[1].sum(
    ), val_data_combined[1].sum(), test_data_combined[1].sum())
    print(train_data_combined[0].size(), train_data_combined[1].size(),
          val_data_combined[0].size(), val_data_combined[1].size(),
          test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1].long())
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1].long())
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1].long())

    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)

    #attr_names = train_dataset_obj.params
    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    # "attr": attr_names, #optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def irregularly_sampled_data_gen(n=10, length=20, seed=0):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    obs_values, ground_truth, obs_times = [], [], []
    for i in range(n):
        t1 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t2 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t3 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        a = 10 * np.random.randn()
        b = 10 * np.random.rand()
        f1 = .8 * np.sin(20*(t1+a) + np.sin(20*(t1+a))) + \
            0.01 * np.random.randn()
        f2 = -.5 * np.sin(20*(t2+a + 20) + np.sin(20*(t2+a + 20))
                          ) + 0.01 * np.random.randn()
        f3 = np.sin(12*(t3+b)) + 0.01 * np.random.randn()
        obs_times.append(np.stack((t1, t2, t3), axis=0))
        obs_values.append(np.stack((f1, f2, f3), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        t = np.linspace(0, 1, 100)
        fg1 = .8 * np.sin(20*(t+a) + np.sin(20*(t+a)))
        fg2 = -.5 * np.sin(20*(t+a + 20) + np.sin(20*(t+a + 20)))
        fg3 = np.sin(12*(t+b))
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2, fg3), axis=0))
    return obs_values, ground_truth, obs_times


def sine_wave_data_gen(args, seed=0):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        t = np.sort(np.random.choice(np.linspace(
            0, 1., 101), size=args.length, replace=True))
        b = 10 * np.random.rand()
        f = np.sin(12*(t+b)) + 0.1 * np.random.randn()
        obs_times.append(t)
        obs_values.append(f)
        tc = np.linspace(0, 1, 100)
        fg = np.sin(12*(tc + b))
        ground_truth.append(fg)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def kernel_smoother_data_gen(args, alpha=100., seed=0, ref_points=10):
    np.random.seed(seed)
    obs_values, ground_truth, obs_times = [], [], []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)

        query_points = np.sort(np.random.choice(
            np.linspace(0, 1., 101), size=args.length, replace=True))
        # query_points = np.sort(np.random.uniform(low=0.0, high=1.0, size=args.length))
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        obs_values.append(query_values)
        obs_times.append(query_points)

        query_points = np.linspace(0, 1, 100)
        weights = np.exp(-alpha*(np.expand_dims(query_points,
                                                1) - np.expand_dims(key_points, 0))**2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)

    obs_values = np.array(obs_values)
    obs_times = np.array(obs_times)
    ground_truth = np.array(ground_truth)
    print(obs_values.shape, obs_times.shape, ground_truth.shape)
    mask = np.ones_like(obs_values)
    combined_data = np.concatenate((np.expand_dims(obs_values, axis=2), np.expand_dims(
        mask, axis=2), np.expand_dims(obs_times, axis=2)), axis=2)
    print(combined_data.shape)
    print(combined_data[0])
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": 1,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def get_toy_data(args):
    dim = 3
    obs_values, ground_truth, obs_times = irregularly_sampled_data_gen(
        args.n, args.length)
    obs_times = np.array(obs_times).reshape(args.n, -1)
    obs_values = np.array(obs_values)
    combined_obs_values = np.zeros((args.n, dim, obs_times.shape[-1]))
    mask = np.zeros((args.n, dim, obs_times.shape[-1]))
    for i in range(dim):
        combined_obs_values[:, i, i *
                            args.length: (i+1)*args.length] = obs_values[:, i]
        mask[:, i, i*args.length: (i+1)*args.length] = 1.
    #print(combined_obs_values.shape, mask.shape, obs_times.shape, np.expand_dims(obs_times, axis=1).shape)
    combined_data = np.concatenate(
        (combined_obs_values, mask, np.expand_dims(obs_times, axis=1)), axis=1)
    combined_data = np.transpose(combined_data, (0, 2, 1))
    print(combined_data.shape)
    train_data, test_data = model_selection.train_test_split(combined_data, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print(train_data.shape, test_data.shape)
    train_dataloader = DataLoader(torch.from_numpy(
        train_data).float(), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(torch.from_numpy(
        test_data).float(), batch_size=args.batch_size, shuffle=False)
    data_objects = {"dataset_obj": combined_data,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": dim,
                    "ground_truth": np.array(ground_truth)}
    return data_objects


def compute_pertp_loss(label_predictions, true_label, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    n_traj, n_tp, n_dims = label_predictions.size()
    label_predictions = label_predictions.reshape(n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj * n_tp, n_dims)
    mask = torch.sum(mask, -1) > 0
    mask = mask.reshape(n_traj * n_tp,  1)
    _, true_label = true_label.max(-1)
    ce_loss = criterion(label_predictions, true_label.long())
    ce_loss = ce_loss * mask
    return torch.sum(ce_loss)/mask.sum()


def get_physionet_data_extrap(args, device, q, flag=1):
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + \
            test_dataset_obj[:len(test_dataset_obj)]
    print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset)
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)

    def extrap(test_data):
        enc_test_data = []
        dec_test_data = []
        for (record_id, tt, vals, mask, labels) in test_data:
            midpt = 0
            for tp in tt:
                if tp < 24:
                    midpt += 1
                else:
                    break
            if mask[:midpt].sum() and mask[midpt:].sum():
                enc_test_data.append(
                    (record_id, tt[:midpt], vals[:midpt], mask[:midpt], labels))
                dec_test_data.append(
                    (record_id, tt[midpt:], vals[midpt:], mask[midpt:], labels))
        return enc_test_data, dec_test_data

    enc_train_data, dec_train_data = extrap(train_data)
    enc_test_data, dec_test_data = extrap(test_data)
    enc_train_data_combined = variable_time_collate_fn(
        enc_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_train_data_combined = variable_time_collate_fn(
        dec_train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    enc_test_data_combined = variable_time_collate_fn(
        enc_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    dec_test_data_combined = variable_time_collate_fn(
        dec_test_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
    print(enc_train_data_combined.shape, dec_train_data_combined.shape)
    print(enc_test_data_combined.shape, dec_test_data_combined.shape)

    # keep the timepoints in enc between 0.0 and 0.5
    enc_train_data_combined[:, :, -1] *= 0.5
    enc_test_data_combined[:, :, -1] *= 0.5
    print(enc_train_data_combined[0, :, -1], dec_train_data_combined[0, :, -1])
    enc_train_dataloader = DataLoader(
        enc_train_data_combined, batch_size=batch_size, shuffle=False)
    dec_train_dataloader = DataLoader(
        dec_train_data_combined, batch_size=batch_size, shuffle=False)
    enc_test_dataloader = DataLoader(
        enc_test_data_combined, batch_size=batch_size, shuffle=False)
    dec_test_dataloader = DataLoader(
        dec_test_data_combined, batch_size=batch_size, shuffle=False)

    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj,
                    "enc_train_dataloader": enc_train_dataloader,
                    "enc_test_dataloader": enc_test_dataloader,
                    "dec_train_dataloader": dec_train_dataloader,
                    "dec_test_dataloader": dec_test_dataloader,
                    "input_dim": input_dim,
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional

    return data_objects


def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask
