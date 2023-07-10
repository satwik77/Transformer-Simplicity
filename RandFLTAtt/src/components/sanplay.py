import torch 
from torch import nn
import numpy as np
import ipdb as pdb
import random
import math

from src.components.attention_utils import MultiHeadedAttention
from src.components.transformer_encoder import Encoder, EncoderLayerFFN
from src.components.positional_encodings import 	PositionalEncoding, LearnablePositionalEncoding




class TransformerCLF(nn.Module):
	def __init__(self, ntoken=4, noutputs=1, d_model=4, nhead=1, d_ffn=8, nlayers=1, dropout=0.0, pos_encode_type ='absolute'):
		super(TransformerCLF, self).__init__()
		self.model_type = 'SAN'
		self.pos_encode_type= pos_encode_type
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, 1000.0)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, max_len= 400)

		self.pos_encode = True
		self.pos_mask = False
		self.d_model = d_model

		self.encoder= nn.Embedding(ntoken, d_model)

		self_attn = MultiHeadedAttention(nhead, d_model, dropout)

		feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )
		encoder_layers = EncoderLayerFFN(d_model, self_attn, feedforward, dropout)

		self.transformer=  Encoder(encoder_layers, nlayers)

		self.decoder= nn.Linear(d_model, noutputs, bias=False)
		self.sigmoid = nn.Sigmoid()



	def init_weights(self, weight_init= 10, inp_init = 1.0, dec_init= 1, bias=False):  ### inp_init and dec_init are not used; uncomment lines 46 and 50 if you want to use them

		self.encoder.weight.data.uniform_(-weight_init, weight_init)
		# self.encoder.weight.data.uniform_(-inp_init, inp_init)
		# if self.bias:
		#     self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-weight_init, weight_init)
		# self.decoder.weight.data.uniform_(-dec_init, dec_init)
		if self.pos_encode_type == 'learnable':        
			self.pos_encoder.state_dict()['pe'].uniform_(-weight_init, weight_init)

		if bias:
			for key in self.transformer.state_dict().keys():
				if 'weight' in key or 'bias' in key:
					self.transformer.state_dict()[key].data.uniform_(-weight_init, weight_init)
		else:
			for key in self.transformer.state_dict().keys():
				if 'weight' in key:
					self.transformer.state_dict()[key].data.uniform_(-weight_init, weight_init)
				elif 'bias' in key:
					self.transformer.state_dict()[key].data.zero_()

	
	def init_xavuni(self):
		self.encoder.weight.data.normal_(mean=0, std= 1)

		torch.nn.init.xavier_uniform_(self.decoder.weight.data)

		if self.pos_encode_type == 'learnable':        
			self.pos_encoder.state_dict()['pe'].normal_(mean=0, std=1)
			
		for key in self.transformer.state_dict().keys():
			if 'weight' in key:
				if self.transformer.state_dict()[key].dim()>1:
					torch.nn.init.xavier_uniform_(self.transformer.state_dict()[key].data)
			elif 'bias' in key:
				self.transformer.state_dict()[key].data.zero_()


	def init_gauss_weights(self, std_init= 10, inp_init = 1.0, dec_init= 1, bias= False):
		
		self.encoder.weight.data.normal_(mean=0, std= inp_init)
		self.decoder.weight.data.normal_(mean=0, std= dec_init)
		# self.encoder.weight.data.uniform_(-inp_init, inp_init)
		# if self.bias:
		#     self.decoder.bias.data.zero_()
		# self.decoder.weight.data.uniform_(-dec_init, dec_init)
		if self.pos_encode_type == 'learnable':        
			self.pos_encoder.state_dict()['pe'].uniform_(-inp_init, inp_init)
	
		if bias:
			for key in self.transformer.state_dict().keys():
				if 'weight' in key or 'bias' in key:
					self.transformer.state_dict()[key].data.normal_(mean=0, std= std_init)
		else:
			for key in self.transformer.state_dict().keys():
				if 'weight' in key:
					self.transformer.state_dict()[key].data.normal_(mean=0, std= std_init)
				elif 'bias' in key:
					self.transformer.state_dict()[key].data.zero_()


	def init_xavnormal(self):
		self.encoder.weight.data.normal_(mean=0, std= 1)
		self.decoder.weight.data.normal_(mean=0, std= 1)


		if self.pos_encode_type == 'learnable':        
			self.pos_encoder.state_dict()['pe'].normal_(mean=0, std=1)
			
		for key in self.transformer.state_dict().keys():
			if 'weight' in key:
				# self.transformer.state_dict()[key].data.normal_(mean=0, std= std_init)
				if self.transformer.state_dict()[key].dim()>1:
					torch.nn.init.xavier_normal_(self.transformer.state_dict()[key].data)
			elif 'bias' in key:
				self.transformer.state_dict()[key].data.zero_()

	

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
		# src = self.encoder(src) 
		if self.pos_encode:
			src= self.pos_encoder(src)

		src = src.transpose(0,1)
		output= self.transformer(src, src_mask)
		slots = src.size(1)
		out_flat= output.view(-1, self.d_model)
		out_idxs= [(i*slots)+lengths[i].item() -1 for i in range(len(lengths))]
		output = out_flat[out_idxs]
		decoded = self.decoder(output)
		decoded = self.sigmoid(decoded)
				
		return decoded
