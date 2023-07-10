# import math
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.positional_encodings import  LearnablePositionalEncoding
import pdb

class RNNModel(nn.Module):
	"""Container module with an embedder, a recurrent module, and a classifier."""

	def __init__(self, rnn_type, ntoken, noutputs, ninp, nhid, nlayers, dropout=0.1, tie_weights=False, is_embedding=True):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		if is_embedding:
			self.encoder = nn.Embedding(ntoken, ninp)
		else:
			ninp = ntoken
			self.encoder = nn.Embedding(ntoken, ninp)
			self.encoder.weight.data =torch.eye(ntoken)
			self.encoder.weight.requires_grad = False
		
		self.bi = bidirectional

		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		
		
		self.decoder = nn.Linear(nhid, noutputs)

		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=1)

		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight
		

		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1

		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden, lengths):
		
		lengths = lengths.cpu()
		inp_emb = self.drop(self.encoder(input))


		emb_packed = nn.utils.rnn.pack_padded_sequence(inp_emb, lengths.cpu(), enforce_sorted = False)
		output_packed, hidden = self.rnn(emb_packed, hidden)
		output_padded, _ = nn.utils.rnn.pad_packed_sequence(output_packed)

		output_flat = output_padded.view(-1, self.nhid)
		slots = input.size(1)
		out_idxs= [(lengths[i].item() -1)*slots + i for i in range(len(lengths))]   # Indices of last hidden state
		out_vecs= output_flat[out_idxs]
		output = self.drop(out_vecs)

		decoded = self.decoder(output)
		decoded = self.softmax(decoded)
		return decoded, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
				weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)



