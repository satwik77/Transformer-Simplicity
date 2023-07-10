# import logging
import pdb
import torch
from glob import glob
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

global padsym
padsym = 'p'

def sent_to_idx(voc, sent, max_length):
	padsym=  'p'
	idx_vec = []
	for w in sent:
		try:
			idx = voc.get_id(w)
			idx_vec.append(idx)
		except:
			pdb.set_trace()
	
	idx_vec.append(voc.get_id(padsym))
	idx_vec = pad_seq(idx_vec, max_length+1, voc)
	
	return idx_vec

def sents_to_idx(voc, sents):
	max_length = max([len(s) for s in sents])
	all_indexes= []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length))
	
	all_indexes = torch.tensor(all_indexes, dtype = torch.long)
	return all_indexes


def pad_seq(seq, max_length, voc):
	seq += [voc.get_id(padsym) for i in range(max_length - len(seq))]
	return seq


def idx_to_sent(voc, arr):
	words = []
	for idx in arr:
		words.append(voc.get_word(idx))
	
	return ' '.join(words)


def idxs_to_sent(voc, arrs):
	sents= []
	for arr in arrs:
		sents.append(idx_to_sent(voc, arr))
	
	return sents
