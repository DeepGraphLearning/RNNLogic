import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MLP, self).__init__()

        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

class FuncToNodeSum(nn.Module):
    def __init__(self, vector_dim):
        super(FuncToNodeSum, self).__init__()

        self.vector_dim = vector_dim
        self.layer_norm = nn.LayerNorm(self.vector_dim)
        self.add_model = MLP(self.vector_dim, [self.vector_dim])

        self.eps = 1e-6
    
    def forward(self, A_fn, x_f, b_n):
        device = x_f.device
        batch_size = b_n.max().item() + 1

        degree = A_fn.sum(0) + 1
        weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
        message = x_f.unsqueeze(0)

        weight_zero = weight == 0
        features = (message * weight).sum(1)
        output = self.add_model(features)
        output = self.layer_norm(output)
        output = torch.relu(output)

        return output

class FuncToNode(nn.Module):
    def __init__(self, vector_dim):
        super(FuncToNode, self).__init__()

        self.vector_dim = vector_dim
        self.layer_norm = nn.LayerNorm(self.vector_dim)
        self.add_model = MLP(self.vector_dim * 12, [self.vector_dim])

        self.eps = 1e-6
    
    def forward(self, A_fn, x_f, b_n):
        device = x_f.device
        batch_size = b_n.max().item() + 1

        degree = A_fn.sum(0) + 1
        weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
        message = x_f.unsqueeze(0)

        weight_zero = weight == 0
        sum = (message * weight).sum(1)
        sq_sum = ((message ** 2) * weight).sum(1)
        min = message.expand(weight.size(0), -1, -1).masked_fill(weight_zero, float('inf')).min(1)[0]
        max = message.expand(weight.size(0), -1, -1).masked_fill(weight_zero, float('-inf')).max(1)[0]

        degree_out = degree.unsqueeze(-1)
        mean = sum / degree_out.clamp(min=self.eps)
        sq_mean = sq_sum / degree_out.clamp(min=self.eps)
        std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
        features = torch.cat([mean, min, max, std], dim=-1)

        scale = degree_out.log()
        sum_scale = torch.zeros(batch_size, device=device)
        cn_scale = torch.zeros(batch_size, device=device)
        ones = torch.ones(scale.size(0), device=device)
        sum_scale.scatter_add_(0, b_n, scale.squeeze(-1))
        cn_scale.scatter_add_(0, b_n, ones)
        mean_scale = sum_scale / cn_scale.clamp(min=self.eps)
        scale = scale / mean_scale[b_n].unsqueeze(-1).clamp(min=self.eps)
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=self.eps)], dim=-1)

        update = features.unsqueeze(-1) * scales.unsqueeze(-2)
        update = update.flatten(-2)

        output = self.add_model(update)
        output = self.layer_norm(output)
        output = torch.relu(output)

        return output