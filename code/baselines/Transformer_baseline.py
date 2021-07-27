"""
Transformer baseline for P12 mortality data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_baselines import *


def number_parameters(model):
    """
    Print the number of model's parameters (trainable and all).

    :param model: model instance
    :return: None
    """
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: ', f'{total_parameters:,}')
    print('Trainable parameters: ', f'{trainable_parameters:,}')


def positional_encoding(X_features, d_model, max_len, X_time):
    """
    Function for positional encoding from the paper "Attention is all you need".

    :param X_features: time series features for all samples
    :param d_model: dimension of a single sequential instance (number of parameters per instance)
    :param max_len: maximum length of the sequence (time series)
    :param X_time: times, when observations were measured; size = (batch, seq_len, 1)
    :return: X_feature with positional encoding (retruned as tensor)
    """
    X_features = torch.from_numpy(X_features)
    for i, sample in enumerate(X_features):
        pe = torch.zeros(max_len, d_model)
        positions = X_time[i]
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(positions / div_term)
        pe[:, 1::2] = torch.cos(positions / div_term)
        X_features[i] += pe
    return X_features


class PositionalEncoding(nn.Module):
    """
    Implement positional encoding following the paper "Attention is all you need" for P12 dataset.
    """
    def __init__(self, d_model, max_len, dropout=0.1):
        """
        Initialize the instance - prepare PE matrix and set dropout rate.

        :param d_model: dimension of a single sequential instance (number of parameters per instance)
        :param max_len: maximum length of the sequence (time series)
        :param dropout: dropout rate
        """
        super().__init__()

        # prepare PE matrix
        self.pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # here we lose the notion of exact time observations
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        self.pe[:, 0::2] = torch.sin(positions / div_term)
        self.pe[:, 1::2] = torch.cos(positions / div_term)

        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_enc', self.pe)

    def forward(self, x):
        x = x + self.pe   # concatenate input with positional encodings
        return self.dropout(x)


class Transformer_P12(nn.Module):
    """
    Transformer model (only encoder part) for time series classification of P12 dataset.
    """
    def __init__(self, d_model, max_len, n_heads, dim_feedforward, activation='relu', dropout=0.1):
        """
        Initialize the model instance.

        :param d_model: dimension of a single sequential instance (number of parameters per instance)
        :param max_len: maximum length of the sequence (time series)
        :param n_heads: number of attention heads
        :param dim_feedforward: dimension of the feedforward network model
        :param activation: activation function of intermediate layer
        :param dropout: dropout rate
        """
        super().__init__()

        # # lose the notion of exact time observations, positional encoding done beforehand
        # self.pos_enc = PositionalEncoding(d_model, max_len)

        self.max_len = max_len

        # starting dropout from positional encoding
        self.pos_enc_dropout = nn.Dropout(p=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, activation=activation, batch_first=True)

        self.feed_forward = nn.Sequential(
            nn.Linear((d_model * max_len) + 9, dim_feedforward),   # (215 * 36) + 9 (static features) = 7749
            nn.ReLU(),
            nn.Linear(dim_feedforward, 2)
            # nn.Softmax(dim=1)   # the softmax function is already included in cross-entropy loss
        )

        # self.init_weights()

    def init_weights(self):
        init_range = 0.01
        self.encoder_layer.weight.data.uniform_(-init_range, init_range)

    def create_padding_mask(self, X_time):
        """
        Create padding mask of the size (batch, seq_len).

        :param X_time: times, when observations were measured; size = (batch, seq_len, 1)
        :return: return the BoolTensor mask with the size (batch, seq_len)
        """
        time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for
                       times in X_time]  # list, len(time_length)=len(X_time)
        mask = torch.zeros([len(X_time), self.max_len])
        for i in range(len(X_time)):
            mask[i, time_length[i]:] = torch.ones(max_len - time_length[i])
        mask = mask.type(torch.BoolTensor)
        return mask

    def forward(self, X_features, X_static, X_time):
        """
        Feed-forward process of the network.

        :param X_features: input of time series features (with already added positional encoding)
        :param X_static: input of static features
        :param X_time: times, when observations were measured; size = (batch, seq_len, 1)
        :return: binary values at the output layer
        """
        # create mask for zero padding
        mask = self.create_padding_mask(X_time)

        # apply dropout for output from positional encoding
        x = self.pos_enc_dropout(X_features)

        # pass through transformer encoder layer
        x = self.encoder_layer(x, src_key_padding_mask=mask)

        # concatenate static features to the flattened encoder layer
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = torch.cat((x, X_static), dim=1)

        # pass through fully-connected part to lower dimension to 2 (binary classification)
        return self.feed_forward(x)


def train_model(model, X_train, X_val, num_epochs, learning_rate, batch_size):
    """
    Train the model.

    :param model: model instance
    :param X_train: (X_features_train, X_static_train, X_time_train, y_train)
    :param X_val: (X_features_val, X_static_val, X_time_val, y_val)
    :param num_epochs: number of epochs
    :param learning_rate: learning rate for optimizer
    :param batch_size: batch size
    :return: None
    """
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_features_train, X_static_train, X_time_train, y_train = X_train
    X_features_val, X_static_val, X_time_val, y_val = X_val

    print('\n------------------\nTraining started\n------------------')

    for epoch in range(num_epochs):
        model.train()   # set model to the training mode

        # todo: batch training - DataLoader?

        # forward pass

        y_pred = model(X_features_val, X_static_val, X_time_val)   # todo: to train

        # compute and print loss
        loss = loss_fun(y_pred, y_val)  # todo: to train

        print('Epoch %d: loss: %.3f' % (epoch, loss.item()))

        # zero out all of the gradients
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # make an update to model parameters
        optimizer.step()

        # check validation set
        model.eval()    # set model to the evaluation mode
        if epoch % 10 == 0:
            with torch.no_grad():
                y_pred = model(X_features_val, X_static_val, X_time_val)
                loss = loss_fun(y_pred, y_val)

                # compute classification accuracy

                # compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)

                # compute average precision and area under precision-recall curve


    print('\n------------------\nTraining finished\n------------------')



if __name__ == '__main__':
    base_path = '../../P12data'
    split_idx = 1
    split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'

    normalization = True
    imputation_method = None  # possible values: None, 'mean', 'forward', 'kNN', 'MICE', 'CubicSpline'

    (X_features_train, X_static_train, X_time_train, y_train), (X_features_val, X_static_val, X_time_val, y_val), (X_features_test, X_static_test, X_time_test, y_test) = read_and_prepare_data(base_path, split_path, normalization, imputation=imputation_method)

    d_model = 36    # number of features per time step
    max_len = 215   # max length of time series
    n_heads = 2     # number of heads does not change the number of model parameters
    dim_feedforward = 256

    # apply positional encoding to the input
    X_features_train = positional_encoding(X_features_train, d_model, max_len, X_time_train)
    X_features_val = positional_encoding(X_features_val, d_model, max_len, X_time_val)
    X_features_test = positional_encoding(X_features_test, d_model, max_len, X_time_test)
    print(X_features_train.shape, X_features_val.shape, X_features_test.shape)

    model = Transformer_P12(d_model, max_len, n_heads, dim_feedforward)
    model = model.double()
    # model.cuda()
    number_parameters(model)
    print(model)

    # X_features_val = X_features_val.cuda()
    # X_static_val = X_static_val.cuda()
    # X_time_val = X_time_val.cuda()

    # pred = model(X_features_val, X_static_val, X_time_val)
    # print('output shape: ', pred.shape, pred[0], pred[1])

    num_epochs = 20
    batch_size = 32
    learning_rate = 0.0001
    X_train = (X_features_train, X_static_train, X_time_train, y_train)
    X_val = (X_features_val, X_static_val, X_time_val, y_val)

    train_model(model, X_train, X_val, num_epochs, learning_rate, batch_size)










