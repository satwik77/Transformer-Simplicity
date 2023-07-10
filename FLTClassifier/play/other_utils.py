import numpy as np
import torch
import pickle
from attrdict import AttrDict
import ipdb as pdb
import random


def gen_rand_binstr(length=40):
    return ''.join([random.choice(['1','0']) for _ in range(length)])


def sample_bstr(data_size = 1000, length=40):
    lines= set()
    size=0
    while size < data_size:
        inp = gen_rand_binstr(length=40)
        lines.add(inp)
        size= len(lines)
    
    lines = list(lines)
    return lines


def flip_i(bstr, idx):
    lstr = list(bstr)
    if lstr[idx] == '0':
        lstr[idx] ='1'
    else:
        lstr[idx] = '0'
    
    return ''.join(lstr)



def hamming_one(binstr):
    binlist= list(binstr)
    hamming_str= []
    for i in range(len(binlist)):
        newlist = binlist.copy()
        if newlist[i] == '0':
            newlist[i] = '1'
        else:
            newlist[i] = '0'
        
        hamming_str.append(''.join(newlist))
    
    return hamming_str


def hamming_two(binstr):
    ham_str =[]
    one_ham = hamming_one(binstr)
    for bstr in one_ham:
        ham_str += hamming_one(bstr)
    
    ham_str = list(set(ham_str))
    ham_str.remove(binstr)
    return ham_str





class Vocab:
    def __init__(self):
        # self.w2id= {'</s>': 0, '1':1, '0':2 }
        # self.id2w = {0:'</s>', 1:'1', 2: '0'}
        # self.w2id= {'1':1, '0':0 }
        # self.id2w = { 1:'1', 0: '0'}
        self.w2id= {'1':1, '0':0 , 'p':2}
        self.id2w = { 1:'1', 0: '0', 2:'p'}
    
    def get_id(self, word):
        return self.w2id[word]
    
    def get_word(self, idx):
        return self.id2w[idx]
    
    def sent2idx(self, sent):
        ids = []
        for j in sent:
            ids.append(self.get_id(j))
        
        return ids



global padsym
padsym=  'p'
def sent_to_idx(voc, sent, max_length):
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


