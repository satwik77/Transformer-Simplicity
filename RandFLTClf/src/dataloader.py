import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict
from src.utils.sentence_processing import sents_to_idx, idxs_to_sent
import pickle



class Corpus(object):
	def __init__(self, path, voc, debug=False):
		self.voc = voc
		self.debug= debug
		self.data, self.targets= self.create_ids(path)
		self.nlabels = max(self.targets)+1
		

	def create_ids(self, path):
		assert os.path.exists(path)

		
		endsym=  's'

		
		label_tensors = []
		# df = pd.read_csv(path, sep='\t')
		with open(path, 'rb') as handle:
			df= pickle.load(handle)


		lines = df['line']
		lines = [x.strip() for x in lines]   # Adding last symbol for classification
		lines = [list(x) for x in lines]

		labels = df['label']
		label_types= list(set(labels))
		label_types.sort()
		label_keys = OrderedDict()

		for k in range(len(label_types)):
			label_keys[label_types[k]] = k
		
		labels = [label_keys[labels[i]] for i in range(len(labels))]
		label_tensors = torch.tensor(labels).type(torch.int64)

		if self.debug:
			return lines[:100], label_tensors[:100]

		return lines, label_tensors






class Sampler(object):
	def __init__(self, corpus, voc, batch_size):
		self.corpus= corpus
		self.batch_size = batch_size
		self.voc = voc
		self.data =corpus.data
		self.targets = corpus.targets
		self.num_batches = np.ceil(len(self.data)/batch_size)


	def get_batch(self, i):
		batch_size= min(self.batch_size, len(self.data) - i)
		
		word_batch = self.data[i: i+batch_size]
		target_batch = self.targets[i:i+batch_size]

		word_lens= torch.tensor([len(x) for x in word_batch], dtype = torch.long)

		try:
			batch_ids= sents_to_idx(self.voc, word_batch)
		except:
			pdb.set_trace()

		source = batch_ids[:,:-1].transpose(0,1)
		targets= target_batch.clone()
		
		return source, targets, word_lens

	def __len__(self):
		return len(self.data)





