import numpy as np
import torch
import pickle
from attrdict import AttrDict
import ipdb as pdb
import random
from scipy.stats import entropy


def compute_entropy(strlabels):
    labels= list(strlabels.values())
    ones = sum(labels)
    one_prob = ones/len(labels)
    zero_prob= 1-one_prob
    ent = entropy([one_prob, zero_prob], base=2)
    return ent



def compute_sensi(strlabels):
    samples = list(strlabels.keys())
    length = len(samples[0])
    mis_counter = 0.0
    for bstr in samples:
        orig_lab= strlabels[bstr]
        for i in range(length):
            fstr = flip_i(bstr, i)
            new_lab = strlabels[fstr]
            if orig_lab != new_lab:
                mis_counter+=1
    
    ratio = mis_counter/len(samples)
    avg_sensi = ratio/length
    return avg_sensi
                
            
def compute_csr(strlabels):
    samples = list(strlabels.keys())
    length = len(samples[0])
    mis_counter = 0.0
    for bstr in samples:
        orig_lab= strlabels[bstr]
        for i in range(length):
            fstr = flip_i(bstr, i)
            new_lab = strlabels[fstr]
            if orig_lab != new_lab:
                mis_counter+=1
                break
    
    csr = mis_counter/len(samples)
    return csr


def compute_sop(strlabels):
    from sympy.logic import SOPform
    from sympy import symbols
    import string
    samples = list(strlabels.keys())
    length = len(samples[0])
    minterms = []
    for key in strlabels:
        value = strlabels[key]
        if value ==1:
            minterms.append(int(key,2))
    
    allsyms = list(string.ascii_lowercase)
    syms = ' '.join(allsyms[:length])
    
    s = symbols(syms)
    boolexp= SOPform(s, minterms)
    
    if bool(boolexp):
        boolstr = str(boolexp)
        expsize = len(boolstr.split())
        return expsize
    else:
        return 0


def dec_to_bin(x, length= 10):
    binstr= '{0:b}'.format(x)
    pad_len = length-len(binstr)
    fstr = '0'*pad_len + binstr
    assert len(fstr) == length
    return fstr
    

def allbins(length =5):
    assert length <= 15
    samples = [dec_to_bin(x, length) for x in range(2**length)]
    return samples





def gen_rand_binstr(length=40):
    return ''.join([random.choice(['1','0']) for _ in range(length)])


def sample_bstr(data_size = 1000, length=20):
    lines= set()
    size=0
    while size < data_size:
        inp = gen_rand_binstr(length=length)
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

