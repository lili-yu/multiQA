import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import tqdm



class EmbeddingLayer(object):
    def __init__(self, args, vocab, embs, oov='<oov>', pad='<pad>'):

        w2i = {word: index for index, word in enumerate(vocab)}

        custvocab = ['yes', 'no', 'irrelevant']
        for word in custvocab:
            if word not in w2i:
                w2i[word] = len(w2i)


        w2i[pad] = len(w2i)
        w2i[oov] = len(w2i)

        self.w2i = w2i
        self.n_V = len(w2i)
        self.n_d = args.word_vec_size
        self.oovid = w2i[oov]
        self.padid = w2i[pad]

        print('Vocab size = {}, Embedding dimension = {}'.format(self.n_V, self.n_d))

        # Build embedding
        self.embedding = nn.Embedding(self.n_V, self.n_d)
        

        # load embedding:
        if embs is not None:
            print('Loading pretrained weights...')
            embwords, embvecs = embs
            unknown_embedding_indices = []
            #for word, index in tqdm(self.w2i.items(), total=len(self.w2i)):
            for word, index in self.w2i.items():
                if word not in embwords:
                    unknown_embedding_indices.append(index)
                else:
                    self.embedding.weight.data[index].copy_(torch.from_numpy(embvecs[embwords.index(word)]))

        known_embedding_indices = list(set(w2i.values()) - set(unknown_embedding_indices))
        emb_mean = self.embedding.weight.data[known_embedding_indices].mean()
        emb_std = self.embedding.weight.data[known_embedding_indices].std()
        self.embedding.weight.data[unknown_embedding_indices].normal_(emb_mean, emb_std)

        print("pre-trained embedding stats: mean={}  std={}\n".format(emb_mean, emb_std))

        self.embedding.weight.requires_grad = args.train_emb

        if args.train_emb:
            print('Embedding layer will be trained.')
        else:
            print('Embedding layer weight fixed.')


