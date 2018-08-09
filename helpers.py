
import sys
import gzip
import random
import argparse
from collections import namedtuple


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#from model import LSTM, EmbeddingLayer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def load_embedding_npz(path):
    data = np.load(path)
    return [ str(w) for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path) as fin:
        fin.readline()
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = ''.join(parts[:-300])
                vector = parts[-300:]
                words.append(word)
                vals += [ float(x) for x in vector ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)



def pad_batch(sequences, emblayer,  oov='<oov>', convert_to_indices=True, pad_left=False):

    #print(sequences)
    sequences = [[x for ph in seq for x in ph.split() ] for seq in sequences]
    #print(sequences)
    max_len = max(len(seq) for seq in sequences)
    # Pad sequences
    pad = '<pad>' if convert_to_indices else emblayer.padid
    #sequences = [list(seq.numpy()) + [pad] * (max_len - len(seq)) for seq in sequences]
    sequences = [seq + [pad] * (max_len - len(seq)) for seq in sequences]
    # Convert words to indices
    if convert_to_indices:
        word2id, oovid = emblayer.w2i, emblayer.oovid
        sequences = [[word2id.get(word, oovid) for word in seq] for seq in sequences]

    data = torch.LongTensor(sequences)  # (seq_len, batch_size)

    return data

def pad_faq(sequences, emblayer,  pad_left=False):

    max_len = max(len(seq) for seq in sequences)
    # Pad sequences
    pad = emblayer.padid
    #sequences = [list(seq.numpy()) + [pad] * (max_len - len(seq)) for seq in sequences]
    sequences = [seq + [pad] * (max_len - len(seq)) for seq in sequences]
    data = torch.LongTensor(sequences)  # (seq_len, batch_size)
    return data
'''

def convert_to_state(model, args):  
    state = { }
    state['args'] = vars(args)
    state['state_dict'] = {}

    # use copies instead of references
    for k, v in model.state_dict().items():
        state['state_dict'][k] = v.clone()

    emb_state = model.embedding_layer.__dict__.copy()
    del emb_state['embedding']
    state['emb_state'] = emb_state
    return state

def save_checkpoint(filepath, state, current_dev, acc, epoch):
    auc05 = current_dev
    print('saving checkpoint: %s_auc05_%.2f_acc_%.2f_e%d.pt'% (filepath, auc05, acc, epoch))
    torch.save(state,'%s_auc05_%.2f_acc_%.2f_e%d.pt'% (filepath, auc05, acc, epoch) )


def load_checkpoint(ckpt, cuda, **kwargs):
    if cuda:
        state = torch.load(ckpt)
    else:
        # ensure tensors are loaded onto CPU
        state = torch.load(ckpt, map_location=lambda storage, loc: storage)
    #print(state.keys())
    args = argparse.Namespace()
    args.__dict__.update(state['args'])
    args.__dict__.update(kwargs)
    import pprint; pprint.pprint(args.__dict__)

    embedding_layer = EmbeddingLayer(args.word_vec_size,
        ['<s>', '</s>'],
    )
    embedding_layer.__dict__.update(state['emb_state'])
    embedding_layer.embedding = nn.Embedding(
        embedding_layer.n_V,
        embedding_layer.n_d
    )

    model = LSTM(embedding_layer,args)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, args


'''



