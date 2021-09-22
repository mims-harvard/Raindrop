import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import os
# os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
# os.add_dll_directory(os.path.dirname(__file__))


from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

# from torch_geometric.nn import TransformerConv
from transformer_conv import TransformerConv
from Ob_propagation import Observation_progation

from sklearn.metrics.pairwise import cosine_similarity


from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

# """import HGT"""
# from pyHGT.conv import *

# class PositionalEncoding0(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding0, self).__init__()

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         pe = self.pe[:x.size(0), :]
#         return pe

# class PositionalEncoding0(nn.Module):
#     def __init__(self, d_model, max_len=500, MAX=10000):
#         super(PositionalEncoding0, self).__init__()
#         self.max_len = max_len
#         self.d_model = d_model
#         self.MAX     = MAX

#     def getPE(self, P_time):
#         B = P_time.shape[1]
#         position = torch.Tensor(P_time.cpu()).unsqueeze(2) # max_len x B
#         pe = torch.zeros(self.max_len, B, self.d_model)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(self.MAX) / self.d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         return pe

#     def forward(self, P_time):
#         pe = self.getPE(P_time)
#         pe = pe.cuda()
#         return pe

import warnings
import numbers

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,): # the d_model is the output dim of time encoding
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):  # P_time.shape = [215, 128]
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)  # shape: (16,). A sequence from 1  to 215 with 16 elements

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)  # shape: [215, 128, 1]  # max_len x B
        scaled_time = times / torch.Tensor(timescales[None, None, :])  # shape [215, 128, 16]
        """Use a 32-D embedding to represent a single time point."""
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class TransformerModel(nn.Module):
    """ Transformer model with context embedding, aggregation
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    """

    def __init__(self, d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, d_static, MAX, n_classes):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        #         self.pos_encoder = PositionalEncoding0(d_model, max_len, MAX)
        self.pos_encoder = PositionalEncodingTF(d_model, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Linear(d_inp, d_model)
        self.d_model = d_model

        self.emb = nn.Linear(d_static, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]

        src = self.encoder(src) * math.sqrt(self.d_model)

        pe = self.pos_encoder(times)

        src = src + pe

        if static is not None:
            emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        x = torch.cat([emb.unsqueeze(0), src], dim=0)

        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()
        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        # masked aggregation
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        lengths2 = lengths.unsqueeze(1)
        output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)

        # feed through MLP
        output = self.mlp(output)
        return output


class TransformerModel2(nn.Module):
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        MAX  = positional encoder MAX parameter
        n_classes = number of classes
    """

    def __init__(self, d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, d_static, MAX, perc, aggreg, n_classes, static=True):
        super(TransformerModel2, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        d_pe = 16
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_pe+d_enc, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Linear(d_inp, d_enc)
        # self.d_model = d_model

        self.emb = nn.Linear(d_static, d_inp)

        # d_fi = d_pe+d_enc + 16 + d_inp
        d_fi = d_pe+d_enc  + d_inp
        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.Linear(d_fi, n_classes),
        )

        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]

        """Take the observations, not the mask"""
        src = src[:, :, :int(src.shape[2]/2)] # remove the mask info

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) #* math.sqrt(self.d_model)  # linear layer: 72 --> 32
        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        """Here are two options: plus or concat"""
        #         src = src + pe
        src = torch.cat([pe, src], axis=2)  # shape: [215, 128, 64]


        src = self.dropout(src)

        if static is not None:
            emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        # """215-D for time series and 1-D for static info"""
        x = src

        """mask out the all-zero rows. """
        # mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        output = self.transformer_encoder(x, src_key_padding_mask=mask) # output.shape: [216, 128, 64]

        """What if no transformer? just MLP, the performance is still good!!!"""
        # output = x

        # masked aggregation: this really matters.
        mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        # output = torch.sum(output , dim=0) / (lengths.unsqueeze(1) + 1)

        # feed through MLP
        """concat static"""
        if static is not None:
            output = torch.cat([output, emb], dim=1)  # [128, 36*5+9] # emb with dim: d_model
        output = self.mlp(output)  # two linears: 64-->64-->2
        return output, 0, 0

# class HGT_latconcat(nn.Module):
#     ""
#     """Implement the raindrop stratey one by one."""
#     """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
#     Inputs:
#         d_inp = number of input features
#         d_model = number of expected model input features
#         nhead = number of heads in multihead-attention
#         nhid = dimension of feedforward network model
#         dropout = dropout rate (default 0.1)
#         max_len = maximum sequence length
#         MAX  = positional encoder MAX parameter
#         n_classes = number of classes
#     """
#
#     def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
#                  MAX=100, perc=0.5, aggreg='mean', n_classes=2):
#         super(HGT_latconcat, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#
#
#         """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
#             (36, 64, 4, 128, 2, 0.3, 215)
#             d_static, MAX, 0.5, aggreg, n_classes,
#             (9, 100, 0.5, 'mean', 2) """
#
#         d_pe = int(perc * d_model)
#         d_enc = d_model - d_pe
#
#         self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
#
#         encoder_layers = TransformerEncoderLayer(int(d_model/2), nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#
#         """HGT layers"""
#         self.gcs = nn.ModuleList()
#         conv_name = 'dense_hgt' #  'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
#         num_types, num_relations = 36, 1
#         nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5
#         for l in range(nlayers):
#             self.gcs.append(GeneralConv(conv_name, 215, 215, num_types, num_relations, nhead_HGT, dropout,
#                                         use_norm = False, use_RTE = False))
#         self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64) # 2 times fully-connected graph
#         self.adj = Parameter(torch.Tensor(36, 36))
#
#
#         """For GIN"""
#         in_D, hidden_D, out_D = 215, 215, 1  # each timestamp input 1 value, map to 64-D feature, output is 64
#         self.GIN1 = GINConv(
#             Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
#                        Linear(hidden_D, hidden_D), ReLU()))
#         self.GIN2 = GINConv(
#             Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
#                        Linear(hidden_D, hidden_D), ReLU()))
#
#
#
#
#         self.encoder = nn.Linear(d_inp, d_enc)
#         self.d_model = d_model
#
#         self.emb = nn.Linear(d_static, d_model)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, n_classes),
#         )
#
#         self.aggreg = aggreg
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 1e-10
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.emb.weight.data.uniform_(-initrange, initrange)
#         glorot(self.adj)
#         # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5
#
#     def forward(self, src, static, times, lengths):
#         """Input to the model:
#         src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
#         static = Pstatic: [128, 9]: this one doesn't matter; static features
#         times = Ptime: [215, 128]: the timestamps
#        lengths = lengths: [128]: the number of nonzero recordings.
#         """
#         maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]
#
#         """Question: why 72 features (36 feature + 36 mask)?"""
#         src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 72 --> 32
#
#         pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
#         # pe.shape = [215, 128, 32]
#
#         """Use late concat"""
#         src = self.dropout(src)  # [215, 128, 36]
#         emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64
#
#
#
#         # append context on front
#         """215-D for time series and 1-D for static info"""
#         # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
#         # """If don't concat static info:"""
#         x = src  # [215, 128, 36]
#
#
#         """mask out the all-zero rows. """
#         mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
#         mask = mask.squeeze(1).cuda()  # shape: [128, 216]
#
#         """Using non-graph tranformer"""
#         # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
#         # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]
#
#
#         adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2) # keep adj symmetric!
#         edge_index_direct = torch.nonzero(adj).T
#         edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)),dim=0)
#         edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]
#         """Where to add the edge weights???"""
#         # edge_index[0]
#         # source node type: duplicate it for each sample.
#         sample_nodetype =  torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
#         source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)
#
#         """If use HGT: take x and edgeindex as input"""
#         # x = x.permute(1, 2, 0).reshape([-1, 215]) # x: [215, 128, 36] --> [128*36, 215]
#         # for gc in self.gcs:
#         #     # train_dataset = dataset[BATCH_SIZE:]
#         #     # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
#         #     x = gc(x,  edge_index=edge_index, node_type=source_node_type,
#         #            edge_type=self.edge_type_train.to(x.device), edge_time=None)
#
#
#         """using GIN"""
#         x = x.permute(1, 2, 0).reshape([-1, 215])  # x: [215, 128, 36] --> [128*36, 215]
#         x = self.GIN1(x, edge_index)
#         x = self.GIN2(x, edge_index) # output is [128*36, 215]
#
#
#         output = x
#         output = output.reshape([-1, 36, 215]).permute(2, 0,1)    # reshape: [128 * 36, 215] --> [215, 128, 36]
#
#         """Late concat"""
#         output_withtime = torch.cat([pe, output], axis=2)  # shape: [215, 128, 72]
#         output = torch.cat([emb.unsqueeze(0), output_withtime], dim=0)  # x.shape: [216, 128, 72]
#
#
#         # masked aggregation
#         mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
#         if self.aggreg == 'mean':
#             lengths2 = lengths.unsqueeze(1)
#             output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
#         elif self.aggreg == 'max':
#             output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)
#
#         # feed through MLP
#         output = self.mlp(output)  # two linears: 64-->64-->2
#         return output

class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """
    # Originally from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
    # Modified by Xiang

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden  # shape][128, 36]

        # x = x.view(-1, x.size(1)) # x.shape: [36, 215]

        # x = x.reshape([-1, x.size(0)]) # making x.shape: [215, 36]

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

# class LSTM_decomposedGIN(nn.Module):
#     ""
#     """Implement the raindrop stratey one by one."""
#     """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
#     Inputs:
#         d_inp = number of input features
#         d_model = number of expected model input features
#         nhead = number of heads in multihead-attention
#         nhid = dimension of feedforward network model
#         dropout = dropout rate (default 0.1)
#         max_len = maximum sequence length
#         MAX  = positional encoder MAX parameter
#         n_classes = number of classes
#     """
#
#     def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
#                  MAX=100, perc=0.5, aggreg='mean', n_classes=2):
#         super(LSTM_decomposedGIN, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#
#         """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len,
#             (36, 64, 4, 128, 2, 0.3, 215)
#             d_static, MAX, 0.5, aggreg, n_classes,
#             (9, 100, 0.5, 'mean', 2) """
#
#         d_pe = int(perc * d_model)
#         d_enc = d_model - d_pe
#
#         self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
#
#         encoder_layers = TransformerEncoderLayer(int(d_model / 2), nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#
#         """HGT layers"""
#         self.gcs = nn.ModuleList()
#         conv_name = 'dense_hgt'  # 'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
#         num_types, num_relations = 36, 1
#         nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5
#
#         self.edge_type_train = torch.ones([36 * 36 * 2], dtype=torch.int64).cuda()  # 2 times fully-connected graph
#         # self.adj = Parameter(torch.Tensor(36, 36))  # random initialize edges
#
#         self.adj = torch.ones([36, 36]).cuda()  # complete graph
#
#         """For GIN"""
#         in_D, hidden_D, out_D = 1, 1, 1  # each timestamp input 1 value, map to 64-D feature, output is 64
#         self.GINstep1 = GINConv(
#             Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
#                        Linear(hidden_D, hidden_D), ReLU()))
#         self.GIN_middlesteps = GINConv(
#             Sequential(Linear(hidden_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
#                        Linear(hidden_D, hidden_D), ReLU()))
#         self.dim = hidden_D
#
#         self.GINmlp = nn.Sequential(
#             nn.Linear(self.dim + d_static, self.dim + d_static),  # self.dim for observation,d_static for static info
#             nn.ReLU(),
#             nn.Linear(self.dim + d_static, n_classes),
#         )
#
#         """With LSTM"""
#         self.input_dim, self.hidden_dim, self.n_layer = 36, 128, 1
#         self.lstm = LSTMCell(self.input_dim, self.hidden_dim)  # our own LSTM
#
#         # self.lstm_layer = nn.LSTM(  # standard nn LSTM
#         #     input_size=36,
#         #     hidden_size=36,         # LSTM hidden unit
#         #     num_layers=1,           # number of LSTM layer
#         #     bias=True,
#         #     batch_first=False,       # if True input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
#         #     # if False, the input with shape (segment_length, batch, no_feature), which is (215, 128, 36) in our case
#         # )
#
#         d_final = self.hidden_dim + 9  # self.hidden_dim is output of LSTM, 9 is d_static
#         self.mlp_static = nn.Sequential(
#             nn.Linear(d_final, d_final),
#             nn.ReLU(),
#             nn.Linear(d_final, n_classes),
#         )
#
#         #######################
#
#         self.encoder = nn.Linear(d_inp, d_enc)
#         self.d_model = d_model
#
#         self.emb = nn.Linear(d_static, d_model)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, n_classes),
#         )
#
#         self.aggreg = aggreg
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 1e-10
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.emb.weight.data.uniform_(-initrange, initrange)
#         glorot(self.adj)
#         # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5
#
#     def forward(self, src, static, times, lengths):
#         """Input to the model:
#         src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
#         static = Pstatic: [128, 9]: this one doesn't matter; static features
#         times = Ptime: [215, 128]: the timestamps
#        lengths = lengths: [128]: the number of nonzero recordings.
#         """
#         maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]
#
#         """Question: why 72 features (36 feature + 36 mask)?"""
#         src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 72 --> 32
#
#         pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
#         # pe.shape = [215, 128, 32]
#
#         """Use late concat"""
#         src = self.dropout(src)  # [215, 128, 36]
#         emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64
#
#         # append context on front
#         """215-D for time series and 1-D for static info"""
#         # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
#         # """If don't concat static info:"""
#         x = src  # [215, 128, 36]
#
#         """mask out the all-zero rows. """
#         mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
#         mask = mask.squeeze(1).cuda()  # shape: [128, 216]
#
#         # # If use tranformer:
#         # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
#         # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]
#
#         """If use HGT: take x and edgeindex as input"""
#         adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2)  # keep adj symmetric!
#         edge_index_direct = torch.nonzero(adj).T
#         edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)), dim=0)
#         edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]
#         """Where to add the edge weights???"""
#         # edge_index[0]
#         # source node type: duplicate it for each sample.
#         sample_nodetype = torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
#         source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)
#
#         """Using GIN with raindrop"""
#         x = x.permute(1, 2, 0)  # x: [215, 128, 36] --> [128, 36, 215]
#         # x_step1 = x[:, :, 0].reshape([-1, 1])
#         # step_results = self.GINstep1 (x_step1, edge_index=edge_index)
#
#         output = torch.zeros([215, src.shape[1], 36]).cuda()  # shape[215, 128, 36]
#         for stamp in range(0, x.shape[-1]):
#             stepdata = x[:, :, stamp].reshape([-1, 1])  # take [128,36,1 ] as one slice and reshape to [128*36, 1]
#             stepdata = self.GINstep1(stepdata, edge_index=edge_index)
#
#             stepdata = stepdata.reshape([-1, 36]).unsqueeze(0)  # average in the middle dimension
#             output[stamp] = stepdata  # the final output shape is [215, 128, 36] after loop
#
#         """LSTM layer"""
#         # # standard LSTM
#         # r_out, (h_n, h_c) = self.lstm_layer(output.float(), None) # output shape: [215, 128, 36]
#
#         # our own LSTM
#         # output = output.permute(1, 0, 2) # [215, 128, 36] --> [128, 215, 36]
#         h0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
#         c0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
#
#         r_out = torch.zeros([output.shape[0], batch_size, self.hidden_dim]).cuda()  # shape[215, 128, 36]
#         cn = c0[0, :, :]
#         hn = h0[0, :, :]
#         for seq in range(output.shape[0]):  # update in each time step
#             hn, cn = self.lstm(output[seq, :, :], (hn, cn))
#             # r_out.append(hn)
#             r_out[seq] = hn  # the final output shape of r_out is [215, 128, 36]
#
#
#         """ masked aggregation"""
#         mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
#         mask2 = mask2[1:]
#         if self.aggreg == 'mean':
#             lengths2 = lengths.unsqueeze(1)
#             output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
#
#         """concat static"""
#         output = torch.cat([output, static], dim=1)  # [128, 36+9]
#         output = self.mlp_static(output)
#
#
#         return output

class LSTMCell_withtimestamp(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """
    # Originally from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
    # Modified by Xiang

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_withtimestamp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, mask, positional_timestamp):
        hx, cx = hidden  # shape: [128, 36]
        # positional_timestamp.shape: [128, 36]
        x = torch.cat((x, positional_timestamp), dim=1)

        """Masked-LSTM: h_t = m*h_t +(1-m)*h_{t-1}, etc. 
        If mask==0, then the h_t and c_t don't update"""

        # x = x.view(-1, x.size(1)) # x.shape: [36, 215]
        # x = x.reshape([-1, x.size(0)]) # making x.shape: [215, 36]

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))

        """Rain drop"""
        m = mask  # mask.shape: [128, 36]
        # Increase dimension from 1 to n_dim. adjust it by self.hidden_dim in Raindrop class.
        n_dim = int(hy.shape[-1]/m.shape[-1])  # n_dim is the dimension of each node's feature in one time stamp.
        m = torch.repeat_interleave(m, n_dim, dim=1) # repeat each element for n dimes
        cy_mask = m*cy + (1-m)*cx
        hy_mask = m*hy + (1-m)* hx
        return (hy_mask, cy_mask)

        # return (hy, cy)


class Transformer_P12(nn.Module):
    """
    Transformer model (only encoder part) for time series classification of P12 dataset.
    """
    def __init__(self, d_model, max_len, n_heads, dim_feedforward, activation='relu', dropout=0.1):
        """
        Initialize the model instance.

        :param d_model: dimension of a single sequential instance (number of features per instance)
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

        new_d_model = 64
        self.linear_embedding = nn.Linear(d_model, new_d_model)   # embed 36 features into 64 to be consistent with other baselines

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=new_d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, activation=activation, batch_first=True)

        self.static_feed_forward = nn.Linear(9, new_d_model)     # transform 9 static features to longer vector

        self.feed_forward = nn.Sequential(
            nn.Linear(new_d_model, new_d_model),
            nn.ReLU(),
            nn.Linear(new_d_model, 2)
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
        :return: return the BoolTensor mask with the size (batch, seq_len) and time_length
        """
        time_length = [np.where(times == 0)[0][1] if np.where(times == 0)[0][0] == 0 else np.where(times == 0)[0][0] for
                       times in X_time]  # list, len(time_length)=len(X_time)
        mask = torch.zeros([len(X_time), self.max_len])
        for i in range(len(X_time)):
            mask[i, time_length[i]:] = torch.ones(self.max_len - time_length[i])
        mask = mask.type(torch.BoolTensor)
        return mask, time_length

    def forward(self, X_features, X_static, X_time):
        """
        Feed-forward process of the network.

        :param X_features: input of time series features (with already added positional encoding)
        :param X_static: input of static features
        :param X_time: times, when observations were measured; size = (batch, seq_len, 1)
        :return: binary values at the output layer
        """
        # create mask for zero padding
        mask, time_length = self.create_padding_mask(X_time)

        # apply dropout for output from positional encoding
        x = self.pos_enc_dropout(X_features)

        # embed 36 features into 64 to be consistent with other baselines
        x = self.linear_embedding(x)

        # pass through transformer encoder layer
        x = self.encoder_layer(x, src_key_padding_mask=mask)

        # concatenate static features to the matrix (add additional row with the size 64)
        static_x = self.static_feed_forward(X_static).unsqueeze(1)
        x = torch.cat((x, static_x), dim=1)

        # # take the mean of all time steps (rows) with additional row for static features; output size = (batch_size, 64)
        # x = torch.mean(x, 1)

        # take the masked mean of all time steps (rows) with additional row for static features; output size = (batch_size, 64)
        mask = torch.cat((mask, torch.ones((x.size()[0], 1), dtype=torch.bool)), dim=1).unsqueeze(2).long()
        time_length = torch.FloatTensor(time_length).unsqueeze(1)
        x = torch.sum(x * (1 - mask), dim=1) / (time_length + 1)    # masked aggregation

        # pass through fully-connected part to lower dimension to 2 (binary classification)
        return self.feed_forward(x)


class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0,
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0.0, static=True):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size))
        # self.x_mean = torch.autograd.Variable(torch.tensor(x_mean))
        self.x_mean = x_mean.clone().detach().requires_grad_(True)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        ################################
        gate_size = 1  # not used
        ################################

        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mz = torch.nn.Parameter(torch.Tensor(input_size))

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mr = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mh = torch.nn.Parameter(torch.Tensor(input_size))

        # y (output)
        w_hy = torch.nn.Parameter(torch.Tensor(output_size, hidden_size))

        # bias
        b_dg_x = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_y = torch.nn.Parameter(torch.Tensor(output_size))

        layer_params = (w_dg_x, w_dg_h,
                        w_xz, w_hz, w_mz,
                        w_xr, w_hr, w_mr,
                        w_xh, w_hh, w_mh,
                        w_hy,
                        b_dg_x, b_dg_h, b_z, b_r, b_h, b_y)

        param_names = ['weight_dg_x', 'weight_dg_h',
                       'weight_xz', 'weight_hz', 'weight_mz',
                       'weight_xr', 'weight_hr', 'weight_mr',
                       'weight_xh', 'weight_hh', 'weight_mh',
                       'weight_hy']
        if bias:
            param_names += ['bias_dg_x', 'bias_dg_h',
                            'bias_z',
                            'bias_r',
                            'bias_h',
                            'bias_y']

        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """
        Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(GRUD, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(GRUD, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h',
                   'weight_xz', 'weight_hz', 'weight_mz',
                   'weight_xr', 'weight_hr', 'weight_mr',
                   'weight_xh', 'weight_hh', 'weight_mh',
                   'weight_hy',
                   'bias_dg_x', 'bias_dg_h',
                   'bias_z', 'bias_r', 'bias_h', 'bias_y']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def forward(self, input):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        X = torch.squeeze(input[0])  # .size = (33,49)
        Mask = torch.squeeze(input[1])  # .size = (33,49)
        Delta = torch.squeeze(input[2])  # .size = (33,49)
        Hidden_State = torch.autograd.Variable(torch.zeros(self.input_size))

        # step_size = X.size(1)  # 49
        # print('step size : ', step_size)

        output = None
        h = Hidden_State

        # decay rates gamma
        w_dg_x = getattr(self, 'weight_dg_x')
        w_dg_h = getattr(self, 'weight_dg_h')

        # z
        w_xz = getattr(self, 'weight_xz')
        w_hz = getattr(self, 'weight_hz')
        w_mz = getattr(self, 'weight_mz')

        # r
        w_xr = getattr(self, 'weight_xr')
        w_hr = getattr(self, 'weight_hr')
        w_mr = getattr(self, 'weight_mr')

        # h_tilde
        w_xh = getattr(self, 'weight_xh')
        w_hh = getattr(self, 'weight_hh')
        w_mh = getattr(self, 'weight_mh')

        # bias
        b_dg_x = getattr(self, 'bias_dg_x')
        b_dg_h = getattr(self, 'bias_dg_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')

        for layer in range(self.num_layers):

            x = torch.squeeze(X[:, layer:layer + 1])
            m = torch.squeeze(Mask[:, layer:layer + 1])
            d = torch.squeeze(Delta[:, layer:layer + 1])

            # (4)
            gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
            gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

            # (5)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean)

            # (6)
            if self.dropout == 0:
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))

                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''

                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z) * h + z * h_tilde

            else:
                h = gamma_h * h

                z = torch.sigmoid((w_xz * x + w_hz * h + w_mz * m + b_z))
                r = torch.sigmoid((w_xr * x + w_hr * h + w_mr * m + b_r))
                h_tilde = torch.tanh((w_xh * x + w_hh * (r * h) + w_mh * m + b_h))

                h = (1 - z) * h + z * h_tilde

        w_hy = getattr(self, 'weight_hy')
        b_y = getattr(self, 'bias_y')

        output = torch.matmul(w_hy, h) + b_y
        output = torch.sigmoid(output)

        return output


class Simple_classifier(nn.Module):
    ""
    """MLP:36-->32, then concat with positional encoding, masked aggregation; then MLP as classifier """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure = None, static=True):
        super(Simple_classifier, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure

        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 144, 4, 2*d_model, 2, 0.3, 215) 
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = 36 #int(perc * d_model)
        d_enc = 36 #d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model+36, nhead, nhid, dropout) # nhid is the number of hidden, 36 is the dim of timestamp

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.dim = int(d_model / d_inp)

        # d_final = 36*(self.dim+1) + d_model # using transformer in step 3, nhid = 36*4
        if static == False:
            d_final = d_enc + d_pe  # + d_inp  # if static is None
        else:
            d_final = d_enc + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        #######################
        self.d_inp = d_inp
        self.d_model = d_model
        self.encoder = nn.Linear(d_inp, d_enc)
        self.emb = nn.Linear(d_static, d_model)

        self.MLP_replace_transformer = nn.Linear(72, 36)


        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )


        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.emb.weight.data.uniform_(-initrange, initrange)
        # glorot(self.adj)
        # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5

    def forward(self, src, static, times, lengths):
        src =  src[:, :, :self.d_inp]
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 36]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 36 --> 36 # Mapping

        # src = src* math.sqrt(self.d_model)

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.


        """Use late concat for static"""
        src = self.dropout(src)  # [215, 128, 36]
        if static is not None:
            emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        """215-D for time series and 1-D for static info"""




        """step 3: use transformer to aggregate temporal information; batchsize in the middle position"""
        """concat with timestamp"""
        r_out = torch.cat([src, pe], dim=-1)  # output shape: [215, 128, (36*4+36)]

        # r_out = src

        """mask out the all-zero rows. """
        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        masked_agg = True
        if masked_agg ==True:
            """ masked aggregation across rows"""
            # print('masked aggregation across rows')
            mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
            mask2 = mask2[1:]
            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg ==False:
            """Without masked aggregation across rows"""
            output = r_out[-1, :, :].squeeze(0) # take the last step's output, shape[128, 36]

        """concat static"""
        if static is not None:
            output = torch.cat([output, emb], dim=1) # [128, 36*5+9] # emb with dim: d_model
        output = self.mlp_static(output_)  # 45-->45-->2

        return output , 0, output_ # output_ is the learned feature


class Raindrop(nn.Module):
    ""
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure = None, static=True):
        super(Raindrop, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure

        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 144, 4, 2*d_model, 2, 0.3, 215) 
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = 36 #int(perc * d_model)
        d_enc = 36 #d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model+36, nhead, nhid, dropout) # nhid is the number of hidden, 36 is the dim of timestamp

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        """HGT layers"""
        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' #  'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1
        nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5

        self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64).cuda() # 2 times fully-connected graph
        # self.adj = Parameter(torch.Tensor(36, 36))  # random initialize edges
        self.adj = torch.ones([36, 36]).cuda()  # complete graph

        # """For GIN"""
        self.dim = int(d_model/d_inp) # the output dim of each node in graph
        # in_D, hidden_D = 1, self.dim  # each timestamp input 2 value, map to self.dim dimension feature, output is self.dim
        # self.GINstep1 = GINConv(
        #     Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
        #                Linear(hidden_D, hidden_D), ReLU()))


        self.transconv  = TransformerConv(in_channels=36, out_channels=36*self.dim, heads=1) # the head is concated. So the output dimension is out_channels*heads



        # self.GINmlp = nn.Sequential(
        #     nn.Linear(self.dim+d_static, self.dim+d_static),  # self.dim for observation,d_static for static info
        #     nn.ReLU(),
        #     nn.Linear(self.dim+d_static, n_classes),
        # )

        # d_final = 36*(self.dim+1) + d_model # using transformer in step 3, nhid = 36*4
        # d_final = 2*self.node_dim +9  # this is not as good as the previous line
        if static == False:
            d_final = d_enc + d_pe  # + d_inp  # if static is None
        else:
            d_final = d_enc + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        #######################
        self.d_inp = d_inp
        self.d_model = d_model
        self.encoder = nn.Linear(d_inp, d_enc)
        self.emb = nn.Linear(d_static, d_model)

        self.MLP_replace_transformer = nn.Linear(72, 36)


        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )


        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.emb.weight.data.uniform_(-initrange, initrange)
        # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
       lengths = lengths: [128]: the number of nonzero recordings.
        """
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)] # I should use this mask, where to use it?
        src =  src[:, :, :self.d_inp]
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 36]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 36 --> 36
        # src = src* math.sqrt(self.d_model)

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        # """concat with timestamp"""
        # src = torch.cat([src, pe], dim=-1)  # output shape: [215, 128, 36*2]

        """Use late concat for static"""
        src = self.dropout(src)  # [215, 128, 36] # do we really need dropout?
        if static is not None:
            emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        """215-D for time series and 1-D for static info"""
        # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
        # """If don't concat static info:"""

        withmask = False  # False is better
        if withmask==True:
            x = torch.cat([src, missing_mask], dim=-1)
        else:
            x = src  # [215, 128, 36*2]


        """mask out the all-zero rows. """
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None] )
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]


        """If skip step 2, direct process the input data"""
        step2 = True

        if step2 == False:
            output = x
            distance = 0
        elif step2 ==True:

            # # If use tranformer:
            # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
            # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]

            """If use HGT: take x and edgeindex as input"""
            # adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2) # keep adj symmetric! fully connected
            # adj = torch.zeros([36,36]).cuda() # not connected nodes.

            adj = self.global_structure.cuda() # torch.from_numpy(self.global_structure) #.cuda()
            adj[torch.eye(36).byte()] = 1  # add self-loop, set diagonal as one

            # the adjacency is directed.
            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]  # get the edge weights
            # edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)),dim=0)
            # edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]

            # """Where to add the edge weights???"""
            # # source node type: duplicate it for each sample.
            # sample_nodetype =  torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
            # source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)

            """Step 2 (loop for units): graph message passing (ribble)"""
            # x: [215, 128, 36]
            output = torch.zeros([215, src.shape[1] , 36*self.dim]).cuda() # shape[215, 128, 36]
            alpha_all = torch.zeros([edge_index.shape[1],  src.shape[1] ]).cuda()
            for unit in range(0, x.shape[1]):
                """Using transformer conv"""
                # stepdata: [215, 36];      # edge_index: [2, 360]; edge_weights: [360]
                stepdata = x[:, unit, :]
                """Here, the edge_attr is not edge weights, but edge features!
                If we want to the calculation contains edge weights, change the calculation of alpha"""
                # stepdata = stepdata.reshape([-1, 36, 1])
                stepdata, attentionweights = self.transconv(stepdata, edge_index=edge_index, edge_weights=edge_weights,
                                                            edge_attr=None, return_attention_weights=True) #edge_weights
                # stepdata.shape [128, 36*self.dim],
                # attentionweights[0]: edge index, shape:[2, 360], attentionweights[1]: edge weights [360, 1]

                """update edge_index and edge_weights"""
                # edge_weights = attentionweights[1].squeeze(-1) # the edge weights in previous layer is used as the intial edge weights for next layer
                # edge_index = attentionweights[0]

                stepdata = stepdata.reshape([-1, 36*self.dim]).unsqueeze(0)  # average in the middle dimension
                output[:, unit, :] = stepdata # the final output shape is [215, 128, 36*self.dim] after loop
                """update graph structure"""
                # n = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)

                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            #
            # """Step 2: graph message passing (ribble)"""
            # """Using GIN with raindrop"""
            # x = x.permute(1,2, 0)  # x: [215, 128, 36] --> [128, 36, 215]
            #
            # output = torch.zeros([215, src.shape[1] , 36*self.dim]).cuda() # shape[215, 128, 36]
            # for stamp in range(0, x.shape[-1]):
            #     """Use GIN for message passing"""
            #     # """here reshapre to [-1, 2], because input for each node is 2-d: one observation and one timestamp"""
            #     # """TODO (Done): it's better to only use 1-D input (the observation), and concat with timestamp before step 2"""
            #     # stepdata = x[:, :, stamp].reshape([-1, 1])  # take [128,36*2,1 ] as one slice and reshape to [128*36, 2]
            #     # stepdata = self.GINstep1(stepdata, edge_index=edge_index)
            #
            #     """Using transformer conv"""
            #     # stepdata: [128, 36];      # edge_index: [2, 360]; edge_weights: [360]
            #     stepdata = x[:, :, stamp]
            #     """Here, the edge_attr is not edge weights, but edge features!
            #     If we want to the calculation contains edge weights, change the calculation of alpha"""
            #     # stepdata = stepdata.reshape([-1, 36, 1])
            #     stepdata, attentionweights = self.transconv(stepdata, edge_index=edge_index, edge_weights=edge_weights,
            #                                                 edge_attr=None, return_attention_weights=True)
            #     # stepdata.shape [128, 36*self.dim],
            #     # attentionweights[0]: edge index, shape:[2, 360], attentionweights[1]: edge weights [360, 1]
            #     """update edge_index and edge_weights"""
            #     edge_weights = attentionweights[1].squeeze(-1) # the edge weights in previous layer is used as the intial edge weights for next layer
            #     edge_index = attentionweights[0]  #
            #
            #     # observations = stepdata # [128, 36*self.dim]
            #     # cos_sim = torch.zeros([observations.shape[0], 36, 36]).cuda() #
            #     # for row in range(observations.shape[0]):
            #     #     row = observations[row].reshape([4, 36]).cpu().detach().numpy()
            #     #     cos_sim_unit = cosine_similarity(row.T)  # shape: (9590, 9590)
            #     #     cos_sim[row,:, :] = torch.from_numpy(cos_sim_unit).cuda()
            #     #
            #     # ave_sim = torch.mean(cos_sim, dim=0)
            #     # index = torch.argsort(ave_sim, dim=0)
            #     # index_K = index < 10  # is this differentialable?
            #     # updated_adj = index_K * ave_sim  # softmax
            #     # edge_index = torch.nonzero(updated_adj).T.cuda()
            #
            #
            #
            #     """Using HGT layer to replace GIN"""
            #     # x = stepdata.permute(1, 2, 0).reshape([-1, 215]) # x: [215, 128, 36] --> [128*36, 215]
            #     # for gc in self.gcs:
            #     #     # train_dataset = dataset[BATCH_SIZE:]
            #     #     # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            #         # x = gc(x,  edge_index=edge_index, node_type=source_node_type,
            #         #        edge_type=self.edge_type_train.to(x.device), edge_time=None)
            #
            #     stepdata = stepdata.reshape([-1, 36*self.dim]).unsqueeze(0)  # average in the middle dimension
            #     output[stamp] = stepdata # the final output shape is [215, 128, 36*self.dim] after loop
            #     """update graph structure"""

            """calculate Euclidean distance between alphas"""
            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)  # distance shape: [128, 128]
            distance = torch.mean(distance) # the average of all unit pairs


        """step 3: use transformer to aggregate temporal information; batchsize in the middle position"""
        """concat with timestamp"""
        output = torch.cat([output, pe], dim=-1)  # output shape: [215, 128, (36*4+36)]

        step3 = True  # """If also skip step 3 (transformer)"""

        if step3==True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)  # output.shape: [216, 128, 72]
        elif step3==False:
            r_out = output
        #
        # """If also skip step 3 (transformer)"""
        # r_out = output

        # the input and output size of transformer encoder is the same. If want to change it, revise Line 268 in
        # /home/xiangzhang/.local/lib/python3.8/site-packages/torch/nn/modules/transformer.py

        # """LSTM layer"""
        # # # standard LSTM
        # # r_out, (h_n, h_c) = self.lstm_layer(output.float(), None) # output shape: [215, 128, 36]
        #
        # # our own LSTM
        # # output = output.permute(1, 0, 2) # [215, 128, 36] --> [128, 215, 36]
        # h0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
        # c0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
        #
        # r_out = torch.zeros([output.shape[0], batch_size, self.hidden_dim]).cuda() # shape[215, 128, 36]
        # cn = c0[0, :, :]
        # hn = h0[0, :, :]
        #
        # for seq in range(output.shape[0]): # update in each time step
        #     hn, cn = self.lstm(output[seq, :, :], (hn, cn), missing_mask[seq, :, :], pe[seq, :, :])
        #     ### output[seq, :, :].shape:[128, 36]
        #     ### hn.shape = cn.shape = [128,128]
        #     ### missing_mask[seq, :, :].shape: [128, 36] Each value in output has a corresponding mask.
        #     r_out[seq] = hn  # the final output shape of r_out is [215, 128, 36]

        masked_agg = True
        if masked_agg ==True:
            """ masked aggregation across rows"""
            mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg ==False:
            """Without masked aggregation across rows"""
            output = r_out[-1, :, :].squeeze(0) # take the last step's output, shape[128, 36]

            # """pooling function to aggregate node-level features to graph-level features"""
            # # if set self.node_dim as a large number and d_final = 2*self.node_dim +9:
            # output = output.reshape(output.shape[0],  2*self.node_dim, 36)
            # output = torch.mean(output, dim=2)

        """concat static"""
        if static is not None:
            output = torch.cat([output, emb], dim=1) # [128, 36*5+9] # emb with dim: d_model
        output = self.mlp_static(output)  # 45-->45-->2

        return output , distance, None


class Raindrop_v2(nn.Module):
    ""
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure = None, sensor_wise_mask=False, static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 144, 4, 2*d_model, 2, 0.3, 215) 
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = 16 #int(perc * d_model)
        d_enc = 36 #d_model - d_pe

        self.d_inp = d_inp
        self.d_model = d_model
        self.emb = nn.Linear(d_static, d_inp)
        self.d_ob = int(d_model/d_inp)  # dimension of observation embedding

        # self.encoder = nn.Linear(d_inp, d_enc) # in Transformer
        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe,max_len, MAX,)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout) #sensor_wise_mask =True
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout) #self.d_inp*(self.d_ob+16),  d_model+16, nhid is the number of hidden, 16 is the dim of timestamp

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)



        # self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64).cuda() # 2 times fully-connected graph
        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()  # complete graph

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        # self.transconv  = TransformerConv(in_channels=36, out_channels=36*self.dim, heads=1) # the head is concated. So the output dimension is out_channels*heads

        # self.ob_propagation = Observation_progation(in_channels=36*self.d_ob, out_channels=36*self.d_ob, heads=1) # the head is concated. So the output dimension is out_channels*heads

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob) # the head is concated. So the output dimension is out_channels*heads

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)


        if self.sensor_wise_mask ==True:
            d_final = self.d_inp*(self.d_ob+16) + d_inp # d_inp for static attributes
        else:
            d_final = d_model +16 + d_inp # using transformer in step 3, nhid = 36*4
        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        #######################


        # self.MLP_replace_transformer = nn.Linear(72, 36)


        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )


        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)
        # self.R_u.uniform_(-initrange, initrange)  # initialize as 0-0.5

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
       lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]

        """Take the observations, not the mask"""
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)] # I should use this mask, where to use it?
        """Take the observations, not the mask"""
        src = src[:, :, :int(src.shape[2]/2)] # remove the mask info
        n_sensor = self.d_inp


        """Question: why 72 features (36 feature + 36 mask)?"""
        # src = self.encoder(src) # * math.sqrt(self.d_model)  # linear layer: 36 --> 36

        """calculate h_u = \sigma(x_u*R_u)"""
        src = torch.repeat_interleave(src, self.d_ob, dim=-1) # shape: [215, 128, 36] --> [215, 128, 36*4]
        h = F.relu(src*self.R_u)  # [215, 128, 144] # With activation function while self.d_ob >1.
        # # this step increased vector elements, but most are 0 due to activation function.
        # The number of nonzero values from 600 in src to 664 in h

        # h = F.sigmoid(src * self.R_u)

        # h = src*self.R_u  # [215, 128, 144]  # without Relu is better


        # h = self.encoder(src) #* math.sqrt(self.d_model)  # linear layer: 72 --> 32
        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours. pe.shape = [215, 128, 16]
        if static is not None:
            emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        """Use late concat for static"""
        h = self.dropout(h)  # [215, 128, 36] # do we really need dropout?

        # append context on front
        """215-D for time series and 1-D for static info"""
        # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
        # """If don't concat static info:"""

        # withmask = False  # False is better
        # if withmask==True:
        #     x = torch.cat([h, missing_mask], dim=-1)
        # else:
        #     x = h  # [215, 128, 36*2]


        """mask out the all-zero rows. """
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None] )
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        """If skip step 1, direct process the input data"""
        step1 = True

        x=h

        if step1 == False:
            output = x
            distance = 0
        elif step1 ==True:
            # # If use tranformer:
            # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
            # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]

            adj = self.global_structure.cuda() # torch.from_numpy(self.global_structure) #.cuda()
            adj[torch.eye(self.d_inp).byte()] = 1  # add self-loop, set diagonal as one

            # the adjacency is directed.
            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]  # get the edge weights
            # edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)),dim=0)
            # edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]

            """Step 1 (loop for units): graph message passing (ribble). Learn of observation embedding"""
            # x: [215, 128, 36]

            batch_size = src.shape[1]
            n_step = src.shape[0]  # 215 time steps for P12, denotes T.
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda() # shape[215, 128, 36]

            use_beta = False
            if use_beta==True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                """Using transformer conv"""
                # stepdata: [215, 36];      # edge_index: [2, 360]; edge_weights: [360]
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]
                """Here, the edge_attr is not edge weights, but edge features!
                If we want to the calculation contains edge weights, change the calculation of alpha"""
                # stepdata = stepdata.reshape([-1, 36, self.d_ob])  # each sample with shape: [215, 36, 4]

                # stepdata = stepdata.reshape([self.d_inp, stepdata.shape[0]*self.d_ob]) # 36 nodes, each has 215*4 dimension

                """THis reshape is correct!"""
                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0,2)
                stepdata =stepdata.reshape(self.d_inp, n_step*self.d_ob )


                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True) #edge_weights

                """update edge_index and edge_weights"""
                edge_index_layer2 = attentionweights[0] # edge index is not changed in self.ob_propagation. We can find a way  to update it.
                edge_weights_layer2 = attentionweights[1].squeeze(-1) # the edge weights in previous layer is used as the intial edge weights for next ob_propagation layer

                """comment for fast debugging"""
                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                # stepdata.shape [36, 215*self.d_ob],
                # attentionweights[0]: edge index, shape:[2, 360], attentionweights[1]: edge weights [360, 1]



                # stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob]).unsqueeze(0)  # reshape back, first dim is time steps

                """This reshape is correct!"""
                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])


                output[:, unit, :] = stepdata # the final output shape is [215, 128, 36*self.dim] after loop
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            """calculate Euclidean distance between alphas"""
            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)  # distance shape: [128, 128]
            distance = torch.mean(distance) # the average of all unit pairs


        """step 2: use transformer to aggregate temporal information; batchsize in the middle position.
        Learn of sensor embedding. """
        """concat with timestamp"""
        # output = torch.cat([output, pe], axis=2)  # output shape: [215, 128, (36*4+16)]

        if self.sensor_wise_mask==True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob) # [215, 128, 36, 4]
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])  ## [215, 128, 36, 16]
            output = torch.cat([extend_output, extended_pe], dim=-1)  # [215, 128, 36, 20]
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16)) # 16 is time embedding dimension
        else:
            output = torch.cat([output, pe], axis=2)  # output shape: [215, 128, (36*4+16)]

        step2 = True  # """If also skip step 3 (transformer)"""
        if step2==True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)  # output.shape: [216, 128, 72]
        elif step2==False:
            r_out = output

        """Aggregate with sensor-wise mask (missing_mask)"""
        sensor_wise_mask = self.sensor_wise_mask  # False is better than True
        # extended_missing_mask = torch.repeat_interleave(missing_mask, (self.d_ob+16), dim=-1) # [215, 128, 720]


        masked_agg = True
        if masked_agg ==True:
            """ masked aggregation across rows"""
            lengths2 = lengths.unsqueeze(1) # shape[128, 1]
            mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
            if sensor_wise_mask:
                output  = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16)) # [215, 128, 36, 20]
                    out = r_out[:,:, se, :]  # [215, 128, 20]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)  # [128,1]
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:,:, se].unsqueeze(-1)), dim=0) / (len + 1)  # [128, 20]
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg ==False:
            """Without masked aggregation across rows"""
            output = r_out[-1, :, :].squeeze(0) # take the last step's output, shape[128, 36]

            # """pooling function to aggregate node-level features to graph-level features"""
            # # if set self.node_dim as a large number and d_final = 2*self.node_dim +9:
            # output = output.reshape(output.shape[0],  2*self.node_dim, 36)
            # output = torch.mean(output, dim=2)

        """concat static"""
        if static is not None:
            output = torch.cat([output, emb], dim=1)  # [128, 36*5+9] # emb with dim: d_model
        output = self.mlp_static(output)  # 45-->45-->2

        return output , distance, None  # distance is 0 or not


