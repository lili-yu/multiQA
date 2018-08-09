import torch
import torch.nn as nn
import random
import math
import numpy as np

from models.encoder import Encoder
from abc import ABCMeta, abstractmethod
from helpers import pad_faq


class INFOBOT:
    def __init__(self, faqdataset, embedding_layer, args ):
        '''
            INFObot/qbot

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (question).
        '''
        super(INFOBOT, self).__init__()

        self.args = args
        self.tagw2i = faqdataset.tagw2i
        self.tagi2w = faqdataset.tagi2w

        self.tagw2i['STOP'] = len(self.tagw2i )
        self.tagi2w.append('STOP')

        self.faqpool = faqdataset.faqlist
        self.faqnum = len(faqdataset)
        self.actiondim = len(self.tagw2i) #The last action is 'STOP guessing'/'provide faq'

        self.statedim =  300 #args.embedding_dim

        self.hidden_size = self.statedim

        self.state_encoder = Encoder(embedding_layer, args )

        self.policynet = DQN(self.state_encoder, self.statedim, self.actiondim)
        self.targetnet = DQN(self.state_encoder, self.statedim, self.actiondim)

        if args.sharing_encoder:
            self.faq_encoder = self.state_encoder # 
            print('The faq embedding and state encoding are shared')
        else: 
            self.faq_encoder = Encoder(embedding_layer, args)

        self.faqguessed  = 0  #make a change here 
        self.steps = 0
        print('action size: {}'.format(self.actiondim))
        print('Infobot initialized: {} {}'.format(self.faqnum, self.faqpool[1]))



        
        '''
        similarity_measure = args.similarity_measure 
        if similarity_measure == 'cosine':
            self.compute_similarity = self.compute_cosine_similarity(left, right)
        elif similarity_measure == 'dot_product':
            self.compute_similarity = self.compute_dot_product(left, right)
        elif similarity_measure == 'euclidean':
            self.compute_similarity = self.compute_negative_euclidean_distance(left, right)
        '''

    def proceed(self, state, emblayer):
        #state = self.stateencoder(self.history)
        action_index = self._select_action(state)
        action = self.tagi2w[action_index]
        
        '''
        if action == 'STOP':
            self._guessfaq = self._guessfaq()
            self.faqguessed = self._guessfaq
        '''
        self.faqguessed = self._guessfaq(state, emblayer)[0]
        return action_index, action, self.faqguessed



    def _guessfaq(self, state, emblayer):
        state = self.state_encoder(state)  #(1, n_hid)
        faq_embed = self._faq_vect_matrix(emblayer) #(faqn, n_hid)
        faqguessed = self._rank(state, faq_embed)
        return faqguessed

    def _faq_vect_matrix(self, emblayer):
        faqvec = [f['question_idx'] for f in self.faqpool]
        faqvec_mat = []

        batch_size = 200
        num_sample = len(faqvec)
        num_batch = num_sample // batch_size

        for batch_idx in range(num_batch + 1):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            batch = faqvec[start_idx:end_idx]
            if len(batch) == 0:
                break
            batch = pad_faq(batch, emblayer)
            faq_rep = self.faq_encoder(batch)
            faqvec_mat.extend(faq_rep)
        faqvec_mat = torch.stack(faqvec_mat)
        return faqvec_mat


    def _rank(self, statevec, faqvec_mat):

        state_mat = statevec.expand( self.faqnum, -1)

        scores = self.compute_similarity(state_mat, faqvec_mat)
        scores = scores.data.cpu().numpy()
        scores = np.array(scores)
        ranks = scores.argsort()[::-1]
        #new_results = [ whitelist[idx] for idx in ranks ]
        #new_scores = [ scores[idx] for idx in ranks ]
        return ranks

    def _select_action(self, state):
        # Do .detacth() or do ...data()
        self.policynet.eval()
        sample = random.random()
        eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
            math.exp(-1. * self.steps / self.args.eps_decay)
        self.steps +=1
        if sample > eps_threshold:
            with torch.no_grad():
                #values = self.policynet(state.unsqueeze(0))
                values = self.policynet(state)
                action = values.max(1)[1].view(1, 1)
                return action

        else:
            action = torch.tensor([[random.randrange(self.actiondim)]], dtype=torch.long) # device=device,
            return action


    def compute_similarity(self, left, right):
        return torch.bmm(left.unsqueeze(1), right.unsqueeze(2)).squeeze()

    @staticmethod
    def compute_negative_euclidean_distance(left, right):
        return -(F.pairwise_distance(left, right) ** 2)

    @staticmethod
    def compute_dot_product(left, right):
        return torch.bmm(left.unsqueeze(1), right.unsqueeze(2)).squeeze()  # (batch, 1, n_d) x (batch, n_d, 1)

    @staticmethod
    def compute_cosine_similarity(left, right):
        return F.cosine_similarity(left, right, dim=1)


class DQN(nn.Module):
    def __init__(self, encoder, state_dim, num_action):
        super(DQN, self).__init__()
        self.state_encoder = encoder
        self.transform = nn.Linear(state_dim, num_action)

    def forward(self, x):    #x is sequence of tokens
        # x should be sequence of qa pairs
        #print(x.size())
        x = self.state_encoder(x)
        #print(x.size())
        x = self.transform(x)  #(batch, dim, num_action)
        #print(x.size())
        return x

