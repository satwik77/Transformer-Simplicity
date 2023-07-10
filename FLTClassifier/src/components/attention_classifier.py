import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import copy

from src.components.positional_encodings import 	PositionalEncoding, CosineNpiPositionalEncoding, LearnablePositionalEncoding


def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



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
 





class AttCLF(nn.Module):
	def __init__(self, ntoken, noutputs, d_model, d_ffn, nhead=1, dropout=0.25, pos_encode= True, pos_encode_type ='absolute', bias=True):
		super(AttCLF, self).__init__()
		self.model_type = 'SAN'
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, 10000.0)
		elif pos_encode_type == 'cosine_npi':
			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		
		self.pos_encode = pos_encode
		self.pos_mask = False
		self.d_model = d_model

		self.encoder= nn.Embedding(ntoken, d_model)

		self.self_attn = MultiHeadedAttention(nhead, d_model, dropout)

		self.feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )

		self.decoder= nn.Linear(d_model, noutputs, bias=bias)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=1)

		# for p in self.parameters():
		# 	if p.dim() > 1:
		# 		nn.init.xavier_uniform(p)

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		if self.bias:
			self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
	

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask
	

	def forward(self, src, lengths):
		src_mask = None
		if self.pos_mask:
			src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
		
		src = self.encoder(src) * math.sqrt(self.d_model)
		if self.pos_encode:
			src= self.pos_encoder(src)
		
		src = src.transpose(0,1)
		output= self.self_attn(src, src, src, src_mask)
		output = self.feedforward(output)
		
		slots = src.size(1)
		out_flat= output.view(-1, self.d_model)
		out_idxs= [(i*slots)+lengths[i].item() -1 for i in range(len(lengths))]
		out_vecs = out_flat[out_idxs]
		out = self.decoder(out_vecs)
		out = self.softmax(out)

		
		return out
