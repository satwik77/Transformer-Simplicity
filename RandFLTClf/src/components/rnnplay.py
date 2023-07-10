import torch 
from torch import nn
import numpy as np
import ipdb as pdb



class RNNModel(nn.Module):
	def __init__(self, ntoken, noutputs, nemb=10, nhid=10, nlayers=1, rnn_type='LSTM', nonlinearity= 'tanh'):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(0.0)
		self.rnn_type = rnn_type
		self.encoder = nn.Embedding(ntoken, nemb)
		# self.rnn = nn.LSTM(nemb, nhid, nlayers, dropout=0.0, bidirectional=False)
		
		self.decoder= nn.Linear(nhid, noutputs, bias=False)
		# self.softmax = nn.LogSoftmax(dim=1)
		self.sigmoid = nn.Sigmoid()
		
		
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(nemb, nhid, nlayers, dropout=0.0)
		else:
			self.rnn = nn.RNN(nemb, nhid, nlayers, nonlinearity=nonlinearity, dropout=0.0)
		
		# self.rnn_type= 'LSTM'
		# self.init_weights()
		self.nhid= nhid
		self.nlayers =nlayers
	
	def init_weights(self, weight_init= 10.0, inp_init=1.0, dec_init= 1.0, bias=False):
		
		# self.decoder.bias.data.zero_()
		# self.decoder.weight.data.uniform_(-dec_init, dec_init)
		# self.encoder.weight.data.uniform_(-inp_init, inp_init)
		self.decoder.weight.data.uniform_(-weight_init, weight_init)  # inp_init and dec_init have no impact; Use the previous two lines if you want to use them
		self.encoder.weight.data.uniform_(-weight_init, weight_init)

		if bias:
			for key in self.rnn.state_dict().keys():
				if 'weight' in key or 'bias' in key:
					getattr(self.rnn, key).data.uniform_(-weight_init, weight_init)
		else:
			for key in self.rnn.state_dict().keys():
				if 'weight' in key:
					getattr(self.rnn, key).data.uniform_(-weight_init, weight_init)
				elif 'bias' in key:
					self.rnn.state_dict()[key].data.zero_()
	
	def init_xavuni(self, bias=True):
		torch.nn.init.xavier_uniform_(self.decoder.weight.data)
		self.encoder.weight.data.normal_(mean=0, std= 1)

		for key in self.rnn.state_dict().keys():
			if 'weight' in key:
				# getattr(self.rnn, key).data.normal_(mean=0, std= std_init)
				if getattr(self.rnn, key).data.dim()>1:
					torch.nn.init.xavier_uniform_(getattr(self.rnn, key).data)
			elif 'bias' in key:
				self.rnn.state_dict()[key].data.zero_()
		
	
	def init_gauss_weights(self, std_init= 10.0, inp_init=1.0, dec_init= 1.0, bias=False):
		
		# self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-dec_init, dec_init)
		self.encoder.weight.data.uniform_(-inp_init, inp_init)
		
		if bias:
			for key in self.rnn.state_dict().keys():
				if 'weight' in key or 'bias' in key:
					getattr(self.rnn, key).data.normal_(mean=0, std= std_init)
		else:
			for key in self.rnn.state_dict().keys():
				if 'weight' in key:
					getattr(self.rnn, key).data.normal_(mean=0, std= std_init)
				elif 'bias' in key:
					self.rnn.state_dict()[key].data.zero_()
		

	def init_xavnormal(self, bias=True):
		
		torch.nn.init.xavier_normal_(self.decoder.weight.data)
		self.encoder.weight.data.normal_(mean=0, std= 1)


		for key in self.rnn.state_dict().keys():
			if 'weight' in key:
				# getattr(self.rnn, key).data.normal_(mean=0, std= std_init)
				if getattr(self.rnn, key).data.dim()>1:
					torch.nn.init.xavier_normal_(getattr(self.rnn, key).data)
			elif 'bias' in key:
				self.rnn.state_dict()[key].data.zero_()



	def forward(self, inp, hidden, lengths):
		lengths= lengths.cpu()
		emb =self.encoder(inp)
		emb_packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, enforce_sorted=False)
		out_packed, hidden = self.rnn(emb_packed, hidden)
		output_padded, _ = nn.utils.rnn.pad_packed_sequence(out_packed)        
		output_flat = output_padded.view(-1, self.nhid)
		slots = inp.size(1)
		out_idxs= [(lengths[i].item() -1)*slots + i for i in range(len(lengths))]   # Indices of last hidden state
		output = output_flat[out_idxs]

		decoded = self.decoder(output)
		decoded = self.sigmoid(decoded)

		return decoded, hidden
	
	
	
	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
			weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)

