import torch
import torch.nn as nn
import numpy as np

class CBOT:
    # initialize
    def __init__(self, faq, args):
        '''
            customer-Bot Model

            Customer simulation, this will be replaced by a real human during fine-tuning/
            human-in-the-loop stage.

            1st step, offering an initial simple query.
            Clarifying round, read  Q (encoder) from agent_bot and give an answer (decoder??) according to the true FAQ
            When offered an FAQ, give y/n feed back (reward)

        '''
        super(CBOT, self).__init__()

        self.truefaq = int(faq['index']) #faq['feature_vec']
        self.binarytags =faq['binarytag'] # pair of question and answer in dictionary
        self.cattags =faq['catgoricaltag'] # pair of question and answer in dictionary
        self.question =faq['question'] # pair of question and answer in dictionary
        self.catquestions = []

        #self.catquestions = self.posiblecatqs(self.cattags, tagschema)

        self.usingregression = args.reward_shaping
        self.regloss = nn.MSELoss()
        self.turnpenalty = args.turnpenalty
        self.args = args



    def initialq(self,):
        initq = random.sample(self.tags).value()
        return initq


    def proceed(self, question, faqguess):
        if self.args.debug: 
            answer = 'yes'
            #answer = torch.tensor([737])
            reward = 0.001
            return answer, reward

        else:
            reward = -1*self.turnpenalty 
            if question == 'STOP':
                r = self.guess_reward(faqguess)
                if r ==1:
                    print('Bingo')
                reward += r
                answer = '_STOP_'
            else:
                answer = self._giveanswer(question)

            if self.args.reward_shaping: 
                Lregression = self.regloss(faq_feat, self.truefaq)
                reward -= Lregression
            #reward = self.evalfaq(faqguess, self.truefaq)   #This should be really easy to implement
            return answer, reward


    def guess_reward(self, faqguess):
        r = 1.0 if faqguess == self.truefaq else -1.0
        if r ==1.0:
            print('guessing :{}'.format(r))
        return r

    def _giveanswer(self, question):
        # The answer would be [yes, no, irrelevant, some_entity]
        if question in self.catquestions:
            qindex = self.catquestions.index(question)
            answer = self.cattags[qindex]
        elif question in self.binarytags:
            answer = 'yes'
        else: 
            answer = 'no'
        return answer



