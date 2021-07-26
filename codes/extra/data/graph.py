import warnings
import logging
from collections import defaultdict

import torch
# from torch_scatter import scatter_add

from .. import core, utils


logger = logging.getLogger(__name__)


class Graph(core._MetaContainer):
    """
    Basic Graph Data Structure
    """

    meta_types = {"node", "edge", "relation", "graph"}

    def __init__(self, edge_list=None, edge_weight=None, num_node=None, num_relation=None,
                 node_feature=None, edge_feature=None, **kwargs):
        super(Graph, self).__init__(**kwargs)
        # edge_list: N * [h, t] or N * [h, t, r]
        edge_list = self._standarize_edge_list(edge_list, num_relation)
        edge_weight = self._standarize_edge_weight(edge_weight, edge_list)
        num_edge = len(edge_list)

        if num_node is None:
            num_node = self._maybe_num_node(edge_list)
        if num_relation is None and edge_list.shape[1] > 2:
            num_relation = self._maybe_num_relation(edge_list)

        self._edge_list = edge_list
        self._edge_weight = edge_weight
        self.num_node = num_node
        self.num_edge = num_edge
        self.num_relation = num_relation

        if node_feature is not None:
            with self.node():
                self.node_feature = torch.as_tensor(node_feature)
        if edge_feature is not None:
            with self.edge():
                self.edge_feature = torch.as_tensor(edge_feature)

    def node(self):
        return self.context("node")

    def edge(self):
        return self.context("edge")

    def graph(self):
        return self.context("graph")

    def _check_attribute(self, key, value):
        if self._meta_context == "node":
            if len(value) != self.num_node:
                raise ValueError("Expect node attribute `%s` to have shape (%d, *), but found %s" %
                                 (key, self.num_node, value.shape))
        elif self._meta_context == "edge":
            if len(value) != self.num_edge:
                raise ValueError("Expect edge attribute `%s` to have shape (%d, *), but found %s" %
                                 (key, self.num_edge, value.shape))
        return

    def __setattr__(self, key, value):
        self._check_attribute(key, value)
        super(Graph, self).__setattr__(key, value)

    @classmethod
    def from_dense(cls, adjacency, node_feature=None, edge_feature=None):
        """
        Parameters:
            adjacency (array_like): adjacency matrix with shape (num_node, num_node)
            node_feature (array_like): node features with shape (num_node, \*)
            edge_feature (array_like): edge features with shape (num_node, num_node, \*)
        """
        adjacency = torch.as_tensor(adjacency)
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("`adjacency` should be a square matrix, but found %d and %d" % adjacency.shape[:2])

        edge_list = adjacency.nonzero()
        edge_weight = adjacency[tuple(edge_list.t())]
        num_node = adjacency.shape[0]
        num_relation = adjacency.shape[2] if adjacency.ndim > 2 else None
        if edge_feature is not None:
            edge_feature = torch.as_tensor(edge_feature)
            edge_feature = edge_feature[tuple(edge_list.t())]

        return cls(edge_list, edge_weight, num_node, num_relation, node_feature, edge_feature)

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_relation = -1
        data = defaultdict(list)
        meta = graphs[0].meta
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            for k, v in graph.data.items():
                if meta[k] == "graph":
                    v = v.unsqueeze(0)
                data[k].append(v)
            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError("Inconsistent `num_relation` in graphs. Expect %d but got %d."
                                 % (num_relation, graph.num_relation))

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data = {k: torch.cat(v) for k, v in data.items()}

        return PackedGraph(edge_list, edge_weight, num_nodes, num_edges, num_relation,
                           meta=meta, **data)

    def _standarize_edge_list(self, edge_list, num_relation):
        if edge_list is not None and len(edge_list):
            edge_list = torch.as_tensor(edge_list)
        else:
            num_element = 2 if num_relation is None else 3
            edge_list = torch.zeros(0, num_element, dtype=torch.long)
        if edge_list.dtype != torch.long:
            raise ValueError("Can't convert edge_list to torch.long")
        return edge_list

    def _standarize_edge_weight(self, edge_weight, edge_list):
        if edge_weight is not None:
            edge_weight = torch.as_tensor(edge_weight)
            if len(edge_list) != len(edge_weight):
                raise ValueError("`edge_list` and `edge_weight` should be the same size, but found %d and %d"
                                 % (len(edge_list), len(edge_weight)))
        else:
            edge_weight = torch.ones(len(edge_list), device=edge_list.device)
        return edge_weight

    def _maybe_num_node(self, edge_list):
        warnings.warn("_maybe_num_node() is used to determine the number of nodes."
                      "This may underestimate the count if there are isolated nodes.")
        return edge_list[:, :2].max().item() + 1

    def _maybe_num_relation(self, edge_list):
        warnings.warn("_maybe_num_relation() is used to determine the number of relations."
                      "This may underestimate the count if there are unseen relations.")
        return edge_list[:, 2].max().item() + 1

    def _standarize_node_index(self, index):
        if isinstance(index, int):
            index = torch.tensor([index])
        elif isinstance(index, slice):
            start = (index.start + self.num_node) % self.num_node if index.start else 0
            stop = (index.stop + self.num_node) % self.num_node if index.stop else self.num_node
            step = index.step or 1
            index = torch.arange(start, stop, step)
        else:
            index = torch.as_tensor(index)
            if index.dtype == torch.bool:
                index = index.nonzero().squeeze(-1)
        return index

    def get_edge(self, index):
        if len(index) != self.edge_list.shape[1]:
            raise ValueError("Incorrect edge index. Expect %d axes but got %d axes"
                             % (self.edge_list.shape[1], len(index)))

        edge = torch.as_tensor(index, device=self.device)
        edge_index = (self.edge_list == edge).all(dim=-1)
        return self.edge_weight[edge_index].sum()

    def __contains__(self, edge):
        edge = torch.as_tensor(edge, device=self.device)
        return (self.edge_list == edge).any()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = list(index)
        else:
            index = [index]
        while len(index) < self.edge_list.shape[1]:
            index.append(slice(None))

        if all([isinstance(axis_index, int) for axis_index in index]):
            return self.get_edge(index)

        edge_list = self.edge_list.clone()
        for i, axis_index in enumerate(index):
            axis_index = self._standarize_node_index(axis_index)
            mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
            mapping[axis_index] = axis_index
            edge_list[:, i] = mapping[edge_list[:, i]]
        edge_index = (edge_list >= 0).all(dim=-1)

        return self.edge_mask(edge_index)

    def __len__(self):
        return 1

    @property
    def batch_size(self):
        return 1

    def subgraph(self, index):
        return self.node_mask(index, compact=True)

    def data_mask(self, node_index=None, edge_index=None):
        data = {}
        for k, v in self.data.items():
            if self.meta[k] == "node" and node_index is not None:
                v = v[node_index]
            elif self.meta[k] == "edge" and edge_index is not None:
                v = v[edge_index]
            data[k] = v
        return data

    def node_mask(self, index, compact=False):
        index = self._standarize_node_index(index)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[index] = torch.arange(len(index))
            num_node = len(index)
        else:
            mapping[index] = index
            num_node = self.num_node

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)

        if compact:
            data = self.data_mask(index, edge_index)
        else:
            data = self.data_mask(torch.arange(self.num_node), edge_index)

        return type(self)(edge_list[edge_index], self.edge_weight[edge_index], num_node, self.num_relation,
                          meta=self.meta, **data)

    def edge_mask(self, index):
        data = self.data_mask(edge_index=index)

        return type(self)(self.edge_list[index], self.edge_weight[index], self.num_node, self.num_relation,
                          meta=self.meta, **data)

    def full(self):
        index = torch.arange(self.num_node, device=self.device)
        if self.num_relation:
            edge_list = torch.meshgrid(index, index, torch.arange(self.num_relation, device=self.device))
        else:
            edge_list = torch.meshgrid(index, index)
        edge_list = torch.stack(edge_list).flatten(1)
        edge_weight = torch.ones(len(edge_list))

        # remove edge features
        meta = {}
        data = {}
        for key, value in self.meta.items():
            if value != "edge":
                meta[key] = value
                data[key] = getattr(self, key)

        return type(self)(edge_list, edge_weight, self.num_node, self.num_relation, meta=meta, **data)

    @property
    def adjacency(self):
        if not hasattr(self, "_adjacency"):
            if self.num_relation:
                shape = (self.num_node, self.num_node, self.num_relation)
                self._adjacency = torch.sparse_coo_tensor(self.edge_list.t(), self.edge_weight, shape)
            else:
                shape = (self.num_node, self.num_node)
                self._adjacency = torch.sparse_coo_tensor(self.edge_list.t(), self.edge_weight, shape)
        return self._adjacency

    @property
    def node2graph(self):
        return torch.zeros(self.num_node, dtype=torch.long)

    @property
    def edge2graph(self):
        return torch.zeros(self.num_edge, dtype=torch.long)

    @property
    def edge_list(self):
        return self._edge_list

    @property
    def edge_weight(self):
        return self._edge_weight

    @property
    def device(self):
        return self.edge_list.device

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight.cuda(), num_node=self.num_node,
                              num_relation=self.num_relation, meta=self.meta, **utils.cuda(self.data))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight.cpu(), num_node=self.num_node,
                              num_relation=self.num_relation, meta=self.meta, **utils.cpu(self.data))

    def __repr__(self):
        if self.num_relation is None:
            return "%s(num_node=%d, num_edge=%d)" % (self.__class__.__name__, self.num_node, self.num_edge)
        else:
            return "%s(num_node=%d, num_edge=%d, num_relation=%d)" \
                   % (self.__class__.__name__, self.num_node, self.num_edge, self.num_relation)


class PackedGraph(Graph):
    def __init__(self, edge_list=None, edge_weight=None, num_nodes=None, num_edges=None, num_relation=None,
                 offsets=None, **kwargs):
        edge_list, num_nodes, num_edges, num_cum_nodes, num_cum_edges = \
            self._get_cumulative(edge_list, num_nodes, num_edges)

        if offsets is None:
            # special case 1: graphs[-1] has no edge
            graph_index = num_cum_edges < len(edge_list)
            # special case 2: graphs[i] and graphs[i + 1] both have no edge
            offsets = scatter_add(num_nodes[graph_index], num_cum_edges[graph_index], dim_size=len(edge_list))
            offsets = offsets.cumsum(0)
            edge_list = edge_list.clone()
            edge_list[:, :2] += offsets.unsqueeze(-1)

        Graph.__init__(self, edge_list, edge_weight, num_cum_nodes[-1].item(), num_relation, **kwargs)

        self._offsets = offsets
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_cum_nodes = num_cum_nodes
        self.num_cum_edges = num_cum_edges

    def unpack(self):
        graphs = []
        for i in range(len(self.num_nodes)):
            graphs.append(self.get_item(i))
        return graphs

    def unpack_data(self, data, type="auto"):
        if type == "auto":
            if self.num_node == self.num_edge:
                raise ValueError("Ambiguous type. Please specify either `node` or `edge`")
            if len(data) == self.num_node:
                type = "node"
            elif len(data) == self.num_edge:
                type = "edge"
            else:
                raise ValueError("Graph has %d nodes and %d edges, but data has %d entries" %
                                 (self.num_node, self.num_edge, len(data)))
        data_list = []
        if type == "node":
            for i in range(self.num_node):
                data_list.append(data[self.num_cum_nodes[i] - self.num_nodes[i]: self.num_cum_nodes[i]])
        elif type == "edge":
            for i in range(self.num_node):
                data_list.append(data[self.num_cum_edges[i] - self.num_edges[i]: self.num_cum_edges[i]])

        return data_list

    def get_item(self, index):
        node_index = slice(self.num_cum_nodes[index] - self.num_nodes[index], self.num_cum_nodes[index])
        edge_index = slice(self.num_cum_edges[index] - self.num_edges[index], self.num_cum_edges[index])
        edge_list = self.edge_list[edge_index].clone()
        edge_list[:, :2] -= self._offsets[edge_index].unsqueeze(-1)
        data = self.data_mask(node_index, edge_index)

        graph = Graph(edge_list, self.edge_weight[edge_index], self.num_nodes[index], self.num_relation,
                      meta=self.meta, **data)
        return graph

    def _get_cumulative(self, edge_list, num_nodes, num_edges):
        if edge_list is None:
            raise ValueError("`edge_list` should be provided")
        if num_edges is None:
            raise ValueError("`num_edges` should be provided")

        edge_list = torch.as_tensor(edge_list)
        num_edges = torch.as_tensor(num_edges, device=edge_list.device)
        num_cum_edges = num_edges.cumsum(0)
        if num_nodes is None:
            num_nodes = []
            for num_edge, num_cum_edge in zip(num_edges, num_cum_edges):
                num_nodes.append(self._maybe_num_node(edge_list[num_cum_edge - num_edge: num_cum_edge]))
        num_nodes = torch.as_tensor(num_nodes, device=edge_list.device)
        num_cum_nodes = num_nodes.cumsum(0)
        return edge_list, num_nodes, num_edges, num_cum_nodes, num_cum_edges

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)
        if isinstance(index, tuple):
            return self.get_item(index[0])[index[1:]]

    def __len__(self):
        return len(self.num_nodes)

    def full(self):
        # TODO: more efficient implementation?
        graphs = self.unpack()
        graphs = [graph.full() for graph in graphs]
        return graphs[0].pack(graphs)

    @property
    def node2graph(self):
        if not hasattr(self, "_node2graph"):
            # special case 1: graphs[-1] has no node
            node_index = self.num_cum_nodes[self.num_cum_nodes < self.num_node]
            # special case 2: graphs[i] and graphs[i + 1] both have no node
            node2graph = scatter_add(torch.ones_like(node_index), node_index, dim_size=self.num_node)
            self._node2graph = node2graph.cumsum(0)
        return self._node2graph

    @property
    def edge2graph(self):
        if not hasattr(self, "_edge2graph"):
            # special case 1: graphs[-1] has no edge
            edge_index = self.num_cum_edges[self.num_cum_edges < self.num_edge]
            # special case 2: graphs[i] and graphs[i + 1] both have no edge
            edge2graph = scatter_add(torch.ones_like(edge_index), edge_index, dim_size=self.num_edge)
            self._edge2graph = edge2graph.cumsum(0)
        return self._edge2graph

    @property
    def edge2graph(self):
        return torch.zeros(self.num_edge, dtype=torch.long)

    @property
    def batch_size(self):
        return len(self.num_nodes)

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight.cuda(*args, **kwargs),
                              num_nodes=self.num_nodes.cuda(*args, **kwargs),
                              num_edges=self.num_edges.cuda(*args, **kwargs),
                              num_relation=self.num_relation, offsets=self._offsets.cuda(*args, **kwargs),
                              meta=self.meta, **utils.cuda(self.data, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight.cpu(), num_nodes=self.num_nodes.cpu(),
                              num_edges=self.num_edges.cpu(), num_relation=self.num_relation,
                              offsets=self._offsets.cpu(), meta=self.meta, **utils.cpu(self.data))

    def __repr__(self):
        if self.num_relation is None:
            return "%s(batch_size=%d, num_nodes=%s, num_edges=%s)" \
                   % (self.__class__.__name__, self.batch_size, self.num_nodes, self.num_edges)
        else:
            return "%s(batch_size=%d, num_nodes=%s, num_edges=%s, num_relation=%d)" \
                   % (self.__class__.__name__, self.batch_size, self.num_nodes, self.num_edges, self.num_relation)