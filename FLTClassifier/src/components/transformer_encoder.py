import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from src.components.attention_utils import LayerNorm, clones, SublayerConnection


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn"

    def __init__(self, self_attn):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        #self.feed_forward = feed_forward

    def forward(self, x, mask):
        return self.self_attn(x, x, x, mask)

class EncoderLayerFFN(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayerFFN, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model


    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        # return self.feed_forward(self.self_attn(x, x, x, mask))
