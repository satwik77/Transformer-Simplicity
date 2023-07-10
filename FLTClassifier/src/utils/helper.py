import logging
import pdb
import torch
from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import re
try:
	import cPickle as pickle
except ImportError:
	import pickle
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU
	'''
	# torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device


def create_save_directories(log_path, mod_path, res_path):
	'''
		Check if required folders exist or create them
	'''
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	if not os.path.exists(res_path):
		os.makedirs(res_path)
	
	if mod_path:
		if not os.path.exists(mod_path):
			os.makedirs(mod_path)
	

def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)


def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path: including the run_name
			logger variable: to log messages
		Returns:
			checkpoint: path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
	

class Voc:
	def __init__(self):
		self.trimmed = False
		self.frequented = False
		# self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2}
		# self.id2w = {0: '<s>', 1: '</s>', 2: 'unk'}
		# self.w2c = {'unk':1}
		# self.nwords = 3

		self.w2id = { 's': 0, 'p':1}
		self.id2w = {0: 's', 1:'p'}
		self.w2c = {}
		self.nwords = 2

	def add_word(self, word):
		if word not in self.w2id:
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		

	def add_sent(self, sent):
		for word in sent:
			self.add_word(word)


	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader = None, path=None, debug=False):
		if train_dataloader:
			for data in train_dataloader:
				for sent in data['line']:
					self.add_sent(sent)
		elif path:
			with open(path, 'rb') as handle:
				df = pickle.load(handle)
			
			lines = df['line']
			f = [list(x.strip()) for x in lines]
			
			for line in f:
				self.add_sent(line)
					


		# self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords


