    
import sys
import gzip
import random
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import pandas as pd

class Datasetfaq(object):

    def __init__(self, args):
        
        #Read the data and create dataset
        dataraw = self._load_data()
        self.faqlist = self._dataschema1(dataraw)[:50]
        self._buildvocab()
        self.samplefaq()
        #self.action_dim = len(self.tagw2i)


    def samplefaq(self):
        faq = random.choice(self.faqlist)
        print('FAQ sample after processing')
        print(faq)
        print()
        print("There are total {} faqs\n".format(len(self.faqlist)))


    def _buildvocab(self):
        tag_vocab ={w for faq in self.faqlist for seq in faq['binarytag'] for w in seq.split() }
        q_vocab ={w for faq in self.faqlist for w in faq['question'].split() }
        custvocab = set(['yes', 'no', 'irrelevant', 'STOP', '_STOP_'])
        self.vocab = tag_vocab.union(q_vocab).union(custvocab)

        self.tagw2i ={}
        self.tagi2w =[]

        for faq in self.faqlist:
            for tag in faq['binarytag']:
                if tag not in self.tagw2i:
                    self.tagw2i[tag] = len(self.tagw2i)
                    self.tagi2w.append(tag)

    def q_to_index(self, w2i, oovid):
        for faq in self.faqlist:
            text = faq['question']
            text_index =  [w2i.get(word, oovid) for word in text.split()]
            faq.update({'question_idx':text_index})

    def createbatch(self ):
        return 0
        # This is for sampling a batch


    def _load_data(self, path='faq_labelled_samples_V1.csv'):
        df = pd.read_csv('data/faq_labelled_samples_V1.csv').replace(np.nan, '', regex=True)
        print('Loading file from: \n {} \n'.format(path))
        return df



    def _reformfaq(self, faqs):
        newfaqs =[]
        for i, faq in enumerate(faqs): 
            faq_reform ={}

            tl1_list = [x.strip().strip("'") for x in faq['topic_level1'].strip('[]').split(',')]
            tl2_list = [x.strip().strip("'") for x in faq['topic_level2'].strip('[]').split(',')]
            tl3_list = [x.strip().strip("'") for x in faq['topic_level3'].strip('[]').split(',')]
            actionlist = faq["action"].split(',')
            rt = faq["related topic (noun)"].split(',')
            
            faq_reform['question'] = faq['question']
            #binarytag = [x.strip() for x in tl2_list + tl3_list + actionlist + rt]
            faq_reform['binarytag'] = [x.strip() for x in tl2_list + tl3_list + actionlist + rt]
            faq_reform['catgoricaltag'] = [faq['device']] #+ tl1_list 
            faq_reform['index'] = i
            newfaqs.append(faq_reform)        
        return newfaqs


    def _dataschema1(self, df):
        faqlist_all = df[['device', 'topic_level1', 'topic_level2', 'topic_level3',
           'action', 'related topic (noun)', 'title', 'question', 
            'type']].to_dict('records')
        #newfaq =[self._reformfaq(faq) for faq in faqlist_all]
        newfaqs = self._reformfaq(faqlist_all) 
        return newfaqs

    def __iter__(self):
        return self

    __next__ = next

    def __len__(self): 
        return len(self.faqlist)

