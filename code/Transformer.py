import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

"""import HGT"""
from pyHGT.conv import *


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

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
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

        emb = self.emb(static)

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

    def __init__(self, d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, d_static, MAX, perc, aggreg, n_classes):
        super(TransformerModel2, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        d_pe = int(perc * d_model)
        d_enc = d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Linear(d_inp, d_enc)
        self.d_model = d_model

        self.emb = nn.Linear(d_static, d_model)

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

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 72 --> 32

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        """Here are two options: plus or concat"""
        #         src = src + pe
        src = torch.cat([pe, src], axis=2)  # shape: [215, 128, 64]
        src = self.dropout(src)

        emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        """215-D for time series and 1-D for static info"""
        x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]

        """mask out the all-zero rows. """
        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        output = self.transformer_encoder(x, src_key_padding_mask=mask) # output.shape: [216, 128, 64]

        # masked aggregation
        mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        # feed through MLP
        output = self.mlp(output)  # two linears: 64-->64-->2
        return output

class HGT_latconcat(nn.Module):
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
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2):
        super(HGT_latconcat, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'


        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 64, 4, 128, 2, 0.3, 215)
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = int(perc * d_model)
        d_enc = d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(int(d_model/2), nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        """HGT layers"""
        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' #  'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1
        nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5
        for l in range(nlayers):
            self.gcs.append(GeneralConv(conv_name, 215, 215, num_types, num_relations, nhead_HGT, dropout,
                                        use_norm = False, use_RTE = False))
        self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64) # 2 times fully-connected graph
        self.adj = Parameter(torch.Tensor(36, 36))


        """For GIN"""
        in_D, hidden_D, out_D = 215, 215, 1  # each timestamp input 1 value, map to 64-D feature, output is 64
        self.GIN1 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))
        self.GIN2 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))




        self.encoder = nn.Linear(d_inp, d_enc)
        self.d_model = d_model

        self.emb = nn.Linear(d_static, d_model)

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
        glorot(self.adj)
        # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
       lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 72 --> 32

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        """Use late concat"""
        src = self.dropout(src)  # [215, 128, 36]
        emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64



        # append context on front
        """215-D for time series and 1-D for static info"""
        # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
        # """If don't concat static info:"""
        x = src  # [215, 128, 36]


        """mask out the all-zero rows. """
        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        """Using non-graph tranformer"""
        # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
        # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]


        adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2) # keep adj symmetric!
        edge_index_direct = torch.nonzero(adj).T
        edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)),dim=0)
        edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]
        """Where to add the edge weights???"""
        # edge_index[0]
        # source node type: duplicate it for each sample.
        sample_nodetype =  torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
        source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)

        """If use HGT: take x and edgeindex as input"""
        # x = x.permute(1, 2, 0).reshape([-1, 215]) # x: [215, 128, 36] --> [128*36, 215]
        # for gc in self.gcs:
        #     # train_dataset = dataset[BATCH_SIZE:]
        #     # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        #     x = gc(x,  edge_index=edge_index, node_type=source_node_type,
        #            edge_type=self.edge_type_train.to(x.device), edge_time=None)


        """using GIN"""
        x = x.permute(1, 2, 0).reshape([-1, 215])  # x: [215, 128, 36] --> [128*36, 215]
        x = self.GIN1(x, edge_index)
        x = self.GIN2(x, edge_index) # output is [128*36, 215]


        output = x
        output = output.reshape([-1, 36, 215]).permute(2, 0,1)    # reshape: [128 * 36, 215] --> [215, 128, 36]

        """Late concat"""
        output_withtime = torch.cat([pe, output], axis=2)  # shape: [215, 128, 72]
        output = torch.cat([emb.unsqueeze(0), output_withtime], dim=0)  # x.shape: [216, 128, 72]


        # masked aggregation
        mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        # feed through MLP
        output = self.mlp(output)  # two linears: 64-->64-->2
        return output

from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU


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


class LSTM_decomposedGIN(nn.Module):
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
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2):
        super(LSTM_decomposedGIN, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 64, 4, 128, 2, 0.3, 215)
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = int(perc * d_model)
        d_enc = d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(int(d_model / 2), nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        """HGT layers"""
        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt'  # 'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1
        nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5
        for l in range(nlayers):
            self.gcs.append(GeneralConv(conv_name, 215, 215, num_types, num_relations, nhead_HGT, dropout,
                                        use_norm=False, use_RTE=False))
        self.edge_type_train = torch.ones([36 * 36 * 2], dtype=torch.int64).cuda()  # 2 times fully-connected graph
        # self.adj = Parameter(torch.Tensor(36, 36))  # random initialize edges

        self.adj = torch.ones([36, 36]).cuda()  # complete graph

        """For GIN"""
        in_D, hidden_D, out_D = 1, 1, 1  # each timestamp input 1 value, map to 64-D feature, output is 64
        self.GINstep1 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))
        self.GIN_middlesteps = GINConv(
            Sequential(Linear(hidden_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))
        self.dim = hidden_D

        self.GINmlp = nn.Sequential(
            nn.Linear(self.dim + d_static, self.dim + d_static),  # self.dim for observation,d_static for static info
            nn.ReLU(),
            nn.Linear(self.dim + d_static, n_classes),
        )

        """With LSTM"""
        self.input_dim, self.hidden_dim, self.n_layer = 36, 128, 1
        self.lstm = LSTMCell(self.input_dim, self.hidden_dim)  # our own LSTM

        # self.lstm_layer = nn.LSTM(  # standard nn LSTM
        #     input_size=36,
        #     hidden_size=36,         # LSTM hidden unit
        #     num_layers=1,           # number of LSTM layer
        #     bias=True,
        #     batch_first=False,       # if True input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        #     # if False, the input with shape (segment_length, batch, no_feature), which is (215, 128, 36) in our case
        # )

        d_final = self.hidden_dim + 9  # self.hidden_dim is output of LSTM, 9 is d_static
        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        #######################

        self.encoder = nn.Linear(d_inp, d_enc)
        self.d_model = d_model

        self.emb = nn.Linear(d_static, d_model)

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
        glorot(self.adj)
        # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
       lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 72]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 72 --> 32

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        """Use late concat"""
        src = self.dropout(src)  # [215, 128, 36]
        emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64

        # append context on front
        """215-D for time series and 1-D for static info"""
        # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
        # """If don't concat static info:"""
        x = src  # [215, 128, 36]

        """mask out the all-zero rows. """
        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        # # If use tranformer:
        # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
        # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]

        """If use HGT: take x and edgeindex as input"""
        adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2)  # keep adj symmetric!
        edge_index_direct = torch.nonzero(adj).T
        edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)), dim=0)
        edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]
        """Where to add the edge weights???"""
        # edge_index[0]
        # source node type: duplicate it for each sample.
        sample_nodetype = torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
        source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)

        """Using GIN with raindrop"""
        x = x.permute(1, 2, 0)  # x: [215, 128, 36] --> [128, 36, 215]
        # x_step1 = x[:, :, 0].reshape([-1, 1])
        # step_results = self.GINstep1 (x_step1, edge_index=edge_index)

        output = torch.zeros([215, src.shape[1], 36]).cuda()  # shape[215, 128, 36]
        for stamp in range(0, x.shape[-1]):
            stepdata = x[:, :, stamp].reshape([-1, 1])  # take [128,36,1 ] as one slice and reshape to [128*36, 1]
            stepdata = self.GINstep1(stepdata, edge_index=edge_index)

            stepdata = stepdata.reshape([-1, 36]).unsqueeze(0)  # average in the middle dimension
            output[stamp] = stepdata  # the final output shape is [215, 128, 36] after loop

        """LSTM layer"""
        # # standard LSTM
        # r_out, (h_n, h_c) = self.lstm_layer(output.float(), None) # output shape: [215, 128, 36]

        # our own LSTM
        # output = output.permute(1, 0, 2) # [215, 128, 36] --> [128, 215, 36]
        h0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())

        r_out = torch.zeros([output.shape[0], batch_size, self.hidden_dim]).cuda()  # shape[215, 128, 36]
        cn = c0[0, :, :]
        hn = h0[0, :, :]
        for seq in range(output.shape[0]):  # update in each time step
            hn, cn = self.lstm(output[seq, :, :], (hn, cn))
            # r_out.append(hn)
            r_out[seq] = hn  # the final output shape of r_out is [215, 128, 36]


        """ masked aggregation"""
        mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
        mask2 = mask2[1:]
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)

        """concat static"""
        output = torch.cat([output, static], dim=1)  # [128, 36+9]
        output = self.mlp_static(output)


        return output


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
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2):
        super(Raindrop, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'


        """d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
            (36, 64, 4, 128, 2, 0.3, 215)
            d_static, MAX, 0.5, aggreg, n_classes,
            (9, 100, 0.5, 'mean', 2) """

        d_pe = int(perc * d_model)
        d_enc = d_model - d_pe

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(int(d_model/2), nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        """HGT layers"""
        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' #  'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1
        nhead_HGT = 5  # when using HGT, nhead should be times of max_len (i.e., feature dimension), so we set it as 5
        for l in range(nlayers):
            self.gcs.append(GeneralConv(conv_name, 215, 215, num_types, num_relations, nhead_HGT, dropout,
                                        use_norm = False, use_RTE = False))
        self.edge_type_train = torch.ones([36*36*2], dtype= torch.int64).cuda() # 2 times fully-connected graph
        # self.adj = Parameter(torch.Tensor(36, 36))  # random initialize edges
        self.adj = torch.ones([36, 36]).cuda()  # complete graph

        """For GIN"""
        in_D, hidden_D, out_D = 1, 1, 1  # each timestamp input 1 value, map to 64-D feature, output is 64
        self.GINstep1 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))
        self.GIN_middlesteps = GINConv(
            Sequential(Linear(hidden_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))
        self.dim = hidden_D

        self.GINmlp = nn.Sequential(
            nn.Linear(self.dim+d_static, self.dim+d_static),  # self.dim for observation,d_static for static info
            nn.ReLU(),
            nn.Linear(self.dim+d_static, n_classes),
        )

        """With LSTM"""
        self.node_dim = 4
        self.input_dim, self.hidden_dim, self.n_layer = 36*2, 36*2*self.node_dim, 1 # here input_dim = 36*2 because we concat timestamp
        self.lstm = LSTMCell_withtimestamp(self.input_dim, self.hidden_dim) # our own LSTM


        # self.lstm_layer = nn.LSTM(  # standard nn LSTM
        #     input_size=36,
        #     hidden_size=36,         # LSTM hidden unit
        #     num_layers=1,           # number of LSTM layer
        #     bias=True,
        #     batch_first=False,       # if True input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        #     # if False, the input with shape (segment_length, batch, no_feature), which is (215, 128, 36) in our case
        # )


        d_final = self.hidden_dim + d_model #9  # self.hidden_dim is output of LSTM, 9 is d_static
        # d_final = 2*self.node_dim +9  # this is not as good as the previous line
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
        glorot(self.adj)
        # self.adj.uniform_(0, 0.5)  # initialize as 0-0.5



    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
       lengths = lengths: [128]: the number of nonzero recordings.
        """
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src =  src[:, :, :self.d_inp]

        maxlen, batch_size = src.shape[0], src.shape[1]  # src.shape = [215, 128, 36]

        """Question: why 72 features (36 feature + 36 mask)?"""
        src = self.encoder(src) * math.sqrt(self.d_model)  # linear layer: 36 --> 36

        pe = self.pos_encoder(times)  # times.shape = [215, 128], the values are hours.
        # pe.shape = [215, 128, 32]

        """Use late concat"""
        src = self.dropout(src)  # [215, 128, 36]
        emb = self.emb(static)  # emb.shape = [128, 64]. Linear layer: 9--> 64


        # append context on front
        """215-D for time series and 1-D for static info"""
        # x = torch.cat([emb.unsqueeze(0), src], dim=0)  # x.shape: [216, 128, 64]
        # """If don't concat static info:"""
        x = src  # [215, 128, 36]


        """mask out the all-zero rows. """
        mask = torch.arange(maxlen + 1)[None, :] >= (lengths.cpu()[:, None] + 1)
        mask = mask.squeeze(1).cuda()  # shape: [128, 216]

        # # If use tranformer:
        # # use mask[:, 1:] to transfer it from [128, 216] to [128, 215]
        # output = self.transformer_encoder(x, src_key_padding_mask=mask[:, 1:]) # output.shape: [216, 128, 64]

        """If use HGT: take x and edgeindex as input"""
        adj = self.adj.triu() + self.adj.triu(1).transpose(-1, -2) # keep adj symmetric!
        edge_index_direct = torch.nonzero(adj).T
        edge_index_inverse = torch.cat((edge_index_direct[1].unsqueeze(0), edge_index_direct[0].unsqueeze(0)),dim=0)
        edge_index = torch.cat((edge_index_direct, edge_index_inverse), dim=1)  # shape: [2, 2592]
        """Where to add the edge weights???"""
        # source node type: duplicate it for each sample.
        sample_nodetype =  torch.range(0, 35, dtype=torch.int64)  # node types are from 0 to 35
        source_node_type = sample_nodetype.repeat(x.shape[1]).to(x.device)

        """Using HGT layer to replace GIN"""




        """Using GIN with raindrop"""
        x = x.permute(1,2, 0)  # x: [215, 128, 36] --> [128, 36, 215]

        output = torch.zeros([215, src.shape[1] , 36]).cuda() # shape[215, 128, 36]
        for stamp in range(0, x.shape[-1]):
            stepdata = x[:, :, stamp].reshape([-1, 1])  # take [128,36,1 ] as one slice and reshape to [128*36, 1]
            stepdata = self.GINstep1(stepdata, edge_index=edge_index)

            stepdata = stepdata.reshape([-1, 36]).unsqueeze(0)  # average in the middle dimension
            output[stamp] = stepdata # the final output shape is [215, 128, 36] after loop

        """LSTM layer"""
        # # standard LSTM
        # r_out, (h_n, h_c) = self.lstm_layer(output.float(), None) # output shape: [215, 128, 36]

        # our own LSTM
        # output = output.permute(1, 0, 2) # [215, 128, 36] --> [128, 215, 36]
        h0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_dim).cuda())

        r_out = torch.zeros([output.shape[0], batch_size, self.hidden_dim]).cuda() # shape[215, 128, 36]
        cn = c0[0, :, :]
        hn = h0[0, :, :]


        for seq in range(output.shape[0]): # update in each time step
            hn, cn = self.lstm(output[seq, :, :], (hn, cn), missing_mask[seq, :, :], pe[seq, :, :])
            ### output[seq, :, :].shape:[128, 36]
            ### hn.shape = cn.shape = [128,128]
            ### missing_mask[seq, :, :].shape: [128, 36] Each value in output has a corresponding mask.
            r_out[seq] = hn  # the final output shape of r_out is [215, 128, 36]



        # output = r_out[-1, :, :].squeeze(0) # take the last step's output, shape[128, 36]
        masked_agg = False
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

            # """pooling function to aggregate node-level features to graph-level features"""
            # # if set self.node_dim as a large number and d_final = 2*self.node_dim +9:
            # output = output.reshape(output.shape[0],  2*self.node_dim, 36)
            # output = torch.mean(output, dim=2)



        """concat static"""
        output = torch.cat([output, emb], dim=1) # [128, 36+9] # emb with dim: d_model
        output = self.mlp_static(output)


        # # output = stepdata  # shape[128*36, self.dim]
        # # # global pooling, make the output with shape [128, 36] # this 36 means the dimension in GIN
        # # # ###x = global_add_pool(x, batch)  # not easy to build batch variable
        # # output = output.reshape([128, 36, self.dim]) # average in the middle dimension
        #
        # # output = torch.mean(output , dim=1).squeeze(1)
        # # """placeholder to add static info"""
        # # output = torch.cat([output, static], dim=1)  # shape: [ 128, self.dim] --> [128, self.dim+9]
        # # # feed through MLP
        # # output = self.GINmlp(output)  # two linears: self.dim-->self.dim-->2
        #
        # """Late concat"""
        # output_withtime = torch.cat([pe, output], axis=2)  # shape: [215, 128, 72]
        # output = torch.cat([emb.unsqueeze(0), output_withtime], dim=0)  # x.shape: [216, 128, 72]
        # # masked aggregation
        # mask2 = mask.permute(1, 0).unsqueeze(2).long()  # [216, 128, 1]
        # if self.aggreg == 'mean':
        #     lengths2 = lengths.unsqueeze(1)
        #     output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        # elif self.aggreg == 'max':
        #     output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)
        # # feed through MLP
        # output = self.mlp(output)  # two linears: 64-->64-->2

        return output