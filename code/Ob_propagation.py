from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset
from torch.nn import init

import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter, segment_csr


class Observation_progation(MessagePassing):

    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                 n_nodes: int, ob_dim: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1],  heads * out_channels*8) # increase dense to 32-D (32 = 16*2)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))  #w_v dim =16, p_t dim = 16

        self.ob_dim = ob_dim
        self.index = None


        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        # glorot(self.bias)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, edge_weights=None, use_beta=False,
                edge_attr: OptTensor = None, return_attention_weights=None):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta


        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)
        # output.shape: [215, 1, 144], middle is head



        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index  # already updated/pruned in propagate function.

        if self.concat:  # this is True
            out = out.view(-1, self.heads * self.out_channels)  # shape:[215, 144]
        else:
            out = out.mean(dim=1)

        # if self.root_weight:  # this is True, this makes the performance worse
        #     x_r = self.lin_skip(x[1])
        #     if self.lin_beta is not None: # False
        #         beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
        #         beta = beta.sigmoid()
        #         out = beta * x_r + (1 - beta) * out
        #     else:
        #         out += x_r

        if isinstance(return_attention_weights, bool): # True
            assert alpha is not None
            if isinstance(edge_index, Tensor): # True
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message_selfattention(self, x_i: Tensor, x_j: Tensor,edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """x_i and x_j shape: [360, 36], why?"""

        """x_i and x_j shape:  [360, 36, 4]"""
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)  # self.heads = 1
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels) # self-attention weight
        if edge_weights is not None:
            """Multiply with the edge weights"""
            # alpha = alpha*(edge_weights.unsqueeze(-1))
            alpha = edge_weights.unsqueeze(-1)



        alpha = softmax(alpha, index, ptr, size_i)  # This alpha is based on edges. Each edge has an attention weights
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels) # Value
        # if edge_attr is not None:
        #     out += edge_attr

        out *= alpha.view(-1, self.heads, 1)  # alpha*value
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """x_i and x_j shape:  [360, 36, 4]"""
        """here still has x, x=(x_i, x_j)"""
        """x_i and x_j shape:  [360, 215*4]""" # [360, 215*4]
        """x_i: source node; x_j: target node"""

        # query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)  # self.heads = 1
        # key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        # alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)  # self-attention weight
        #
        use_beta = self.use_beta
        """calculate beta attention weights"""
        if use_beta == True:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1,n_step,  32)  # source node feature, after increase dim: [360, 215, 32]
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)# shape [360, 1, 16]
            # w_v_repeat = w_v.repeat(1, n_step, 1,) # [360, 215, 16]

            p_emb = self.p_t.unsqueeze(0) # [1, 215, 16]

            # p_emb_repeat = p_emb.repeat(n_edges,1, 1)
            aa = torch.cat([w_v.repeat(1, n_step, 1,), p_emb.repeat(n_edges,1, 1)], dim=-1) #[360, 215, 32]
            beta = torch.mean(h_W * aa, dim=-1)  # shape [360, 215]


        if edge_weights is not None:
            """Multiply with the edge weights"""
            if use_beta ==True:
                gamma = beta*(edge_weights.unsqueeze(-1)) # shape [360, 215]
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)  # shape[360, 215*4]

                """edge prune, prune out half of edges """
                all_edge_weights = torch.mean(gamma, dim=1)  # [360]
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]  # the index of top K edge weights
                gamma = gamma[index_top_edges]  # shape [180, 215*4]
                self.edge_index = self.edge_index[:, index_top_edges]  # shape[2, 180]
                index = self.edge_index[0]  # update the index which is used for softmax normalization
                x_i = x_i[index_top_edges]  # update the source node

            else:
                gamma = edge_weights.unsqueeze(-1) #* alpha

        self.index = index
        if use_beta==True:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)  # This alpha is based on edges. Each edge has an attention weights

        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        # out = self.lin_value(x_i).view(-1, self.heads, self.out_channels)  # Value

        """This weights can be further decomposed, this will ask for a lot of memory. 
        If memory is not enough, replace it by previous line."""
        # out.shape: [360, 1, 36]
        decompose = False
        if decompose==False:
            out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)  # Value Good
        else:
            """decompose W into w_u * w_v^T. select vectors from self.nodewise_weights based on index"""
            source_nodes = self.edge_index[0]
            target_nodes = self.edge_index[1]
            w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)  # shape [360, 144, 1]
            w2 = self.nodewise_weights[target_nodes].unsqueeze(1)  # shape [360, 1, 144,]
            # # out = torch.bmm(out, torch.bmm(w1, w2))
            out = torch.bmm(x_i.view(-1, self.heads, self.out_channels), torch.bmm(w1, w2))

        if use_beta==True:
            out = out* gamma.view(-1, self.heads, out.shape[-1])  # alpha*value
        else:
            out = out * gamma.view(-1, self.heads, 1)
        # out = x_i
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
