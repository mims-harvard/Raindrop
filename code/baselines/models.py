import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint

import os
os.add_dll_directory('c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin')
os.add_dll_directory(os.path.dirname(__file__))

from torch.nn.parameter import Parameter
from transformer_conv import TransformerConv
from utils_phy12 import linspace_vector


from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

import warnings
import numbers
from torch.nn.init import xavier_uniform_


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000,):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        P_time = P_time.float()

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)

        scaled_time = times / torch.Tensor(timescales[None, None, :])
        # Use a 32-D embedding to represent a single time point
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

        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        if static == False:
            d_fi = d_enc + d_pe
        else:
            d_fi = d_enc + d_pe + d_inp

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
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]

        src = src[:, :, :int(src.shape[2] / 2)]

        src = self.encoder(src)

        pe = self.pos_encoder(times)
        src = torch.cat([pe, src], axis=2)

        src = self.dropout(src)

        if static is not None:
            emb = self.emb(static)

        x = src

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        output = self.transformer_encoder(x, src_key_padding_mask=mask)

        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if self.aggreg == 'mean':
            lengths2 = lengths.unsqueeze(1)
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths2 + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        # feed through MLP
        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp(output)
        return output


class SEFT(nn.Module):
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

    def __init__(self, d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, d_static, MAX, perc, aggreg, n_classes,
                 static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        d_pe = 16
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_value = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_sensor = PositionalEncodingTF(d_pe, max_len, MAX)

        self.linear_value = nn.Linear(1, 16)
        self.linear_sensor = nn.Linear(1, 16)

        self.d_K = 2 * (d_pe+ 16+16)

        encoder_layers = TransformerEncoderLayer(self.d_K, 1, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        encoder_layers_f_prime = TransformerEncoderLayer(int(self.d_K//2), 1, nhid, dropout)
        self.transformer_encoder_f_prime = TransformerEncoder(encoder_layers_f_prime, 2)

        self.emb = nn.Linear(d_static, 16)

        self.proj_weight = Parameter(torch.Tensor(self.d_K, 128))

        self.lin_map = nn.Linear(self.d_K, 128)

        d_fi = 128 + 16

        if static == False:
            d_fi = 128
        else:
            d_fi = 128 + d_pe
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
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.linear_value.weight.data.uniform_(-initrange, initrange)
        self.linear_sensor.weight.data.uniform_(-initrange, initrange)
        self.lin_map.weight.data.uniform_(-initrange, initrange)
        xavier_uniform_(self.proj_weight)

    def forward(self, src, static, times, lengths):
        maxlen, batch_size = src.shape[0], src.shape[1]

        src = src.permute(1, 0,2)
        fea = src[:, :, :int(src.shape[2]/2)]

        output = torch.zeros((batch_size, self.d_K)).cuda()
        for i in range(batch_size):
            nonzero_index = fea[i].nonzero(as_tuple=False)
            if nonzero_index.shape[0] == 0:
                continue
            values = fea[i][nonzero_index[:,0], nonzero_index[:,1]] # v in SEFT paper
            time_index = nonzero_index[:, 0]
            time_sequence = times[:, i]
            time_points = time_sequence[time_index]  # t in SEFT paper
            pe_ = self.pos_encoder(time_points.unsqueeze(1)).squeeze(1)

            variable = nonzero_index[:, 1]  # the dimensions of variables. The m value in SEFT paper.

            unit = torch.cat([pe_, values.unsqueeze(1), variable.unsqueeze(1)], dim=1)

            variable_ = self.pos_encoder_sensor(variable.unsqueeze(1)).squeeze(1)

            values_ = self.linear_value(values.float().unsqueeze(1)).squeeze(1)

            unit = torch.cat([pe_, values_, variable_], dim=1)

            f_prime = torch.mean(unit, dim=0)

            x = torch.cat([f_prime.repeat(unit.shape[0], 1), unit], dim=1)

            x = x.unsqueeze(1)

            output_unit = x
            output_unit = torch.mean(output_unit, dim=0)
            output[i, :] = output_unit

        output = self.lin_map(output)

        if static is not None:
            emb = self.emb(static)

        # feed through MLP
        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp(output)
        return output


class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0,
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0.0, static=True):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size))
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

        self._all_weights = []

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

        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            with torch.no_grad():
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

    def forward(self, input, dataset_name='P12'):
        X = torch.squeeze(input[0])
        Mask = torch.squeeze(input[1])
        Delta = torch.squeeze(input[2])
        Hidden_State = torch.autograd.Variable(torch.zeros(self.input_size))

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
        if dataset_name == 'P12':
            output = torch.sigmoid(output)

        return output


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

        d_pe = 36
        d_enc = 36

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_model+36, nhead, nhid, dropout)  # nhid is the number of hidden, 36 is the dim of timestamp

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.gcs = nn.ModuleList()
        conv_name = 'dense_hgt' # 'hgt' # 'dense_hgt',  'gcn', 'dense_hgt'
        num_types, num_relations = 36, 1

        self.edge_type_train = torch.ones([36*36*2], dtype=torch.int64).cuda()  # 2 times fully-connected graph
        self.adj = torch.ones([36, 36]).cuda()  # complete graph

        """For GIN"""
        self.dim = int(d_model/d_inp)  # the output dim of each node in graph
        in_D, hidden_D = 1, self.dim  # each timestamp input 2 value, map to self.dim dimension feature, output is self.dim
        self.GINstep1 = GINConv(
            Sequential(Linear(in_D, hidden_D), BatchNorm1d(hidden_D), ReLU(),
                       Linear(hidden_D, hidden_D), ReLU()))

        self.transconv  = TransformerConv(in_channels=36, out_channels=36*self.dim, heads=1)

        self.GINmlp = nn.Sequential(
            nn.Linear(self.dim+d_static, self.dim+d_static),  # self.dim for observation, d_static for static info
            nn.ReLU(),
            nn.Linear(self.dim+d_static, n_classes),
        )

        if static == False:
            d_final = d_enc + d_pe
        else:
            d_final = d_enc + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.d_inp = d_inp
        self.d_model = d_model
        self.encoder = nn.Linear(d_inp, d_enc)
        self.static = static
        if self.static:
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
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        maxlen, batch_size = src.shape[0], src.shape[1]

        src = self.encoder(src) * math.sqrt(self.d_model)

        pe = self.pos_encoder(times)
        src = self.dropout(src)
        if static is not None:
            emb = self.emb(static)

        withmask = False
        if withmask==True:
            x = torch.cat([src, missing_mask], dim=-1)
        else:
            x = src

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None] )
        mask = mask.squeeze(1).cuda()

        step2 = True  # If skip step 2, direct process the input data
        if step2 == False:
            output = x
            distance = 0
        elif step2 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(36).byte()] = 1

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            # graph message passing
            output = torch.zeros([maxlen, src.shape[1], 36*self.dim]).cuda()
            alpha_all = torch.zeros([edge_index.shape[1],  src.shape[1] ]).cuda()
            for unit in range(0, x.shape[1]):
                stepdata = x[:, unit, :]
                stepdata, attentionweights = self.transconv(stepdata, edge_index=edge_index, edge_weights=edge_weights,
                                                            edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.reshape([-1, 36*self.dim]).unsqueeze(0)
                output[:, unit, :] = stepdata

                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            # calculate Euclidean distance between alphas
            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        # use transformer to aggregate temporal information
        output = torch.cat([output, pe], dim=-1)

        step3 = True
        if step3==True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step3==False:
            r_out = output

        masked_agg = True
        if masked_agg ==True:
            # masked aggregation across rows
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if self.aggreg == 'mean':
                lengths2 = lengths.unsqueeze(1)
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg ==False:
            # Without masked aggregation across rows
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output, distance, None



from layer import *
'''Adapted from: https://github.com/nnzhan/MTGNN'''
class MTGNN(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, num_static_features=0, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32,
                 skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3,
                 layer_norm_affline=True):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

        self.num_static_features = num_static_features
        self.mlp_2 = nn.Linear(num_nodes + num_static_features, 2)    # to binary classification
        self.mlp_8 = nn.Linear(num_nodes + num_static_features, 8)    # to classification

    def forward(self, input, input_static, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # add static features and run over classification layer
        x = x.squeeze()
        if self.num_static_features > 0:
            x = torch.cat((x, input_static), dim=1)
            x = self.mlp_2(x)
        else:
            x = self.mlp_8(x)

        return x



'''Adapted from: https://github.com/thuwuyinjun/DGM2'''

class GRU_unit_cluster(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100,
                 device=torch.device("cpu"), use_mask=False, dropout=0.0):
        super(GRU_unit_cluster, self).__init__()

        if update_gate is None:
            if use_mask:
                self.update_gate = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
            else:
                self.update_gate = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            if use_mask:
                self.reset_gate = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
            else:
                self.reset_gate = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                    nn.Sigmoid())
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            if use_mask:
                self.new_state_net = nn.Sequential(
                    nn.Linear(latent_dim + 2 * input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                )
            else:
                self.new_state_net = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, latent_dim),
                    nn.Dropout(p=dropout),
                )
        else:
            self.new_state_net = new_state_net

    def forward(self, y_i, x):
        y_concat = torch.cat([y_i, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        concat = y_i * reset_gate

        concat = torch.cat([concat, x], -1)

        new_probs = self.new_state_net(concat)

        new_y_probs = (1 - update_gate) * new_probs + update_gate * y_i

        assert (not torch.isnan(new_y_probs).any())

        return new_y_probs


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        self.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def init_network_weights(self, net, std=0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        n_dims = first_point.size()[-1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :] - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict,
        n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y


class DGM2_O(nn.Module):

    def __init__(self, latent_dim, input_dim, cluster_num, z0_diffeq_solver=None, z0_dim=None, GRU_update=None,
                 n_gru_units=100, device=torch.device("cpu"), use_sparse=False, dropout=0.0, use_mask=False,
                 use_static=False, num_time_steps_and_static=None):

        super(DGM2_O, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        self.dropout = dropout

        if GRU_update is None:
            self.GRU_update = GRU_unit_cluster(latent_dim, input_dim,
                                               n_units=n_gru_units,
                                               device=device, use_mask=use_mask, dropout=dropout).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.cluster_num = cluster_num
        self.use_sparse = use_sparse
        self.use_mask = use_mask

        self.min_steps = 0.0

        self.extra_info = None

        self.concat_data = True

        if self.concat_data:
            self.infer_emitter_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
                nn.Linear(latent_dim + cluster_num, self.cluster_num),
                nn.Dropout(p=self.dropout)
            )
        else:
            self.infer_emitter_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
                nn.Linear(latent_dim, self.cluster_num),
                nn.Dropout(p=self.dropout)
            )

        self.infer_transfer_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.cluster_num, latent_dim),
            nn.Dropout(p=self.dropout)
        )

        self.decayed_layer = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(1, 1),
            nn.Dropout(p=self.dropout)
        )

        ts, static = num_time_steps_and_static
        if use_static:
            self.mlp = nn.Linear(ts * 10 + static, 2)
        else:
            self.mlp = nn.Linear(ts * 10 + static, 8)

    def forward(self, data, time_steps, static_data, run_backwards=False, save_info=False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:, 0, :].unsqueeze(0)

            all_y_i = self.GRU_update(prev_y, xi)

            all_y_i = F.softmax(all_y_i.unsqueeze(0), -1)

            extra_info = None
        else:
            _, latent_y_states, extra_info = self.run_odernn(
                data, time_steps, run_backwards=run_backwards,
                save_info=save_info)

        if save_info:
            self.extra_info = extra_info

        vec = latent_y_states.squeeze()
        vec = torch.permute(vec, (1, 0, 2))
        vec = torch.reshape(vec, (vec.size()[0], vec.size()[1] * vec.size()[2]))

        if static_data is not None:     # add static data
            vec = torch.cat((vec, static_data), dim=1)

        x = self.mlp(vec)
        return x

    def update_joint_probs(self, n_traj, joint_probs, t, latent_y_states, delta_t, full_curr_rnn_input=None):

        # 		n_traj, n_tp, n_dims = data.size()
        if full_curr_rnn_input is None:
            full_curr_rnn_input = torch.zeros((self.cluster_num, n_traj, self.cluster_num), dtype=torch.float,
                                              device=self.device)

            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((n_traj, self.cluster_num), dtype=torch.float, device=self.device)
                curr_rnn_input[:, k] = 1
                full_curr_rnn_input[k] = curr_rnn_input

        z_t_category_infer_full = self.emit_probs(latent_y_states[t], full_curr_rnn_input, delta_t, t)

        updated_joint_probs = torch.sum(
            z_t_category_infer_full * torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)

        joint_probs_sum = torch.sum(updated_joint_probs)

        return updated_joint_probs

    def emit_probs(self, prev_y_state, prev_y_prob, delta_t, i):

        delta_t = delta_t.to(self.device)

        if len(prev_y_prob.shape) > 2:
            prev_y_state = prev_y_state.repeat(prev_y_prob.shape[0], 1, 1)

        if i > 0:
            delta_t = delta_t.float()
            decayed_weight = torch.exp(-torch.abs(self.decayed_layer(delta_t.view(1, 1))))

            decayed_weight = decayed_weight.view(-1)
        else:
            decayed_weight = 0.5

        if self.concat_data:
            input_z_w = torch.cat([prev_y_prob, prev_y_state], -1)
            prev_y_prob = F.softmax(self.infer_emitter_z(input_z_w), -1)
        else:
            prev_y_prob = F.softmax(self.infer_emitter_z(
                (decayed_weight * self.infer_transfer_z(prev_y_prob) + (1 - decayed_weight) * prev_y_state)), -1)

        return prev_y_prob

    def run_odernn_single_step(self, data, time_steps, full_curr_rnn_input=None,
                               run_backwards=False, save_info=False, prev_y_state=None):

        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        n_traj = data.size()[1]
        if prev_y_state is None:
            prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

        prev_t, t_i = time_steps[0], time_steps[1]

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        if (t_i - prev_t) < self.min_steps:
            time_points = torch.stack((prev_t, t_i))
            inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y_state) * (t_i - prev_t)

            assert (not torch.isnan(inc).any())

            ode_sol = prev_y_state + inc
            ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

            assert (not torch.isnan(ode_sol).any())
        else:
            n_intermediate_tp = max(2, ((t_i - prev_t) / self.min_steps).int())

            time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
            ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

            assert (not torch.isnan(ode_sol).any())

        if torch.mean(ode_sol[:, :, 0, :] - prev_y_state) >= 0.001:
            print("Error: first point of the ODE is not equal to initial value")
            print(torch.mean(ode_sol[:, :, 0, :] - prev_y_state))
            exit()

        yi_ode = ode_sol[:, :, -1, :]

        prev_y_state = self.GRU_update(yi_ode, data)

        xi = data[:, :].unsqueeze(0)

        if save_info:
            d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                 #  					 "yi": yi.detach(),
                 #  					 "yi_std": yi_std.detach(),
                 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
            extra_info.append(d)

        return prev_y_state

    def run_odernn(self, data, time_steps, run_backwards=False, save_info=False, exp_y_states=None):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()

        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)

        prev_y_state = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

        joint_probs = torch.zeros([n_tp, n_traj, self.cluster_num], dtype=torch.float, device=self.device)

        if not run_backwards:
            prev_t, t_i = time_steps[0] - 0.01, time_steps[0]
        else:
            prev_t, t_i = time_steps[-1], time_steps[-1] + 0.01

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = (time_steps[-1] - time_steps[0]) / (len(time_steps) * 0.5)

        self.min_steps = minimum_step

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []

        latent_y_states = []

        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        first_ys = 0

        first_y_state = 0

        count = 0

        for i in time_points_iter:

            if (t_i - prev_t) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y_state) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y_state + inc
                ode_sol = torch.stack((prev_y_state, ode_sol), 2).to(self.device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y_state, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :] - prev_y_state) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y_state))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)

            prev_y_state = self.GRU_update(yi_ode, xi)

            if exp_y_states is not None:
                print(torch.norm(exp_y_states[:, count] - prev_y_state))

            latent_y_states.append(prev_y_state.clone())

            if not run_backwards:
                prev_t, t_i = time_steps[i], time_steps[(i + 1) % time_steps.shape[0]]
            else:
                prev_t, t_i = time_steps[(i - 1)], time_steps[i]

            if save_info:
                d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                     #  					 "yi": yi.detach(),
                     #  					 "yi_std": yi_std.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

            count += 1

        latent_y_states = torch.stack(latent_y_states, 1)

        prev_t, t_i = time_steps[0] - 0.01, time_steps[0]

        if run_backwards:
            latent_y_states = torch.flip(latent_y_states, [1])
        prev_y_prob = torch.zeros((1, n_traj, self.cluster_num)).to(self.device)
        for t in range(latent_y_states.shape[1]):
            prev_y_state = latent_y_states[:, t]

            curr_prob = self.emit_probs(prev_y_state, prev_y_prob, t_i - prev_t, t)

            prev_y_prob = curr_prob

            latent_ys.append(prev_y_prob.clone())

            prev_t, t_i = time_steps[t], time_steps[(t + 1) % time_steps.shape[0]]

        latent_ys = torch.stack(latent_ys, 1)

        return latent_ys, latent_y_states, extra_info


