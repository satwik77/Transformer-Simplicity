from ast import Mult
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import pdb


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""

	def __init__(self, d_model, dropout=0.1):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return self.dropout(sublayer(self.norm(x))) + x


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores= scores.masked_fill(mask ==0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout= 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model %h ==0

        self.d_k = d_model //h

        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn= None
        self.dropout= nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        
        if mask is not None:
            mask= mask.unsqueeze(1)
        
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
    
        x, self.attn = attention(query, key, value, mask= mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
    
    
            
