# -*- coding: utf-8 -*-


import math
from tqdm import tqdm
from os.path import join
import random

import random
import numpy as np



import sys
import argparse
from pprint import pprint
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pdb
import sys

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from datasets import Datasetfaq
from helpers import load_embedding, pad_batch, ReplayMemory, Transition
from models.embedding import EmbeddingLayer
from models.agent_bot import INFOBOT
from models.cust_bot import CBOT




def optimize_model(memory, infobot, writer, args, optimizer, embedding_layer):
    BATCH_SIZE = args.batch_size
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    

    state_batch = pad_batch(batch.state, embedding_layer)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = pad_batch(batch.next_state, embedding_layer)
    #print(action_batch)


    '''
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    '''

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = infobot.policynet(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = infobot.target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = infobot.targetnet(next_state_batch).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    if infobot.steps % args.print_every ==0: 
        print('Loss : {}'.format(loss.item()))

    writer.add_scalar('train_loss', loss.item(), infobot.steps)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()



def main(args):

    log_path = join(args.tffolder, args.save.split('/')[-1])

    # Set up Tensorboard writer
    print('tensorboard path is: {}'.format(log_path))
    writer = SummaryWriter(log_path, comment="RL")
    writer.add_text('Args', str(args))

    # Load data   =====
    faqdata = Datasetfaq(args)

    # Load/build model and embeddings
    if args.ckpt:
        print("Loading model from '{}'...\n".format(args.ckpt))
        model, _ = load_checkpoint(args)
    else:
        embs = load_embedding(args.embedding_path) if args.embedding_path else None

        embedding_layer = EmbeddingLayer(args, faqdata.vocab, embs)
        infobot = INFOBOT(faqdata, embedding_layer, args)
        
    faqdata.q_to_index(embedding_layer.w2i, embedding_layer.oovid)
    model = infobot.policynet


    # Move to gpu
    #if args.cuda:
    if not args.no_cuda: 
        print('Put models into cuda\n')
        model.cuda()
    print("{}\n".format(model))
    print('Number of parameters: {:,}\n\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr,
                           betas=[args.adam_beta1, args.adam_beta2], eps=args.adam_eps, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'val':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_gamma, patience=args.patience)

    memory = ReplayMemory(10000)


    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` Tensor. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.
    #
    # Below, `num_episodes` is set small. You should download
    # the notebook and run lot more epsiodes.
    #

    num_episodes = 10000

    for episode in range(num_episodes):
        batch = faqdata.faqlist
        max_guess = 5
        numRounds = 100
        for round in range(numRounds):
            faqtrue = random.choice(batch)
            cbot = CBOT (faqtrue, args)
            #inita = cbot.sampletag()

            numRollout = 9

            # A-Bot and Q-Bot interacting in RL rounds
            state = ['device']
            #state = torch.tensor(state)
            for i in range(max_guess):
                # Run one round of conversation
                
                #statev = torch.cat(state)
                statev = pad_batch([state], embedding_layer)
                action_t, q, faqguess = infobot.proceed(statev, embedding_layer)
                a, reward = cbot.proceed(q, faqguess)
                if i == max_guess-1:
                    reward = cbot.guess_reward(faqguess)

                
                reward = torch.tensor([reward]) #, device=device)
                #next_statev = torch.cat(next_state)

                #next_state = state + q.split() +[a] 
                next_state = state + [q, a]
                memory.push(state , action_t, next_state, reward)


                # Move to the next state
                state = next_state
                    
                # Perform one step of the optimization (on the target network)
                optimize_model( memory, infobot, writer, args, optimizer, embedding_layer)

        # Update the target network
        if episode % args.target_update == 0:
            infobot.targetnet.load_state_dict(infobot.policynet.state_dict())

    writer.close()

    ######################################################################


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler = 'resolve')


    # ========= Computation setting 
    argparser.add_argument("-no_cuda", action='store_true', default=False, help='Turn off cuda')

    # ========= File I/O
    argparser.add_argument("-save", type=str, default='./ckpt/.pt', help="file path to save model checkpoint")
    argparser.add_argument("-data_dir", type=str, default='', help="training data directory")
    argparser.add_argument("-file", type=str, default='', help="training file")
    argparser.add_argument("-tffolder", type=str, default='./tensorboard', help="where to save tensorboard file")

    argparser.add_argument("-save_every", type=int, default=5, help="every * epoch to save pt file")
    argparser.add_argument("-print_every", type=int, default=100, help="every * epoch to print results")
    argparser.add_argument("-small_data", action='store_true', default=False,
                           help='Load small about of data for debugging')

    # ========= Training initialization

    argparser.add_argument("-embedding_path",  type=str,
                           default='/Volumes/DATA-1/train_embeddings/spear.v2.lower.vec300.min2.it5.neg15.thread12.txt',
                           help="path of embedding")
    argparser.add_argument("-embed_coding", type=str, default='avg', choices=['seq', 'avg', 'id', 'bow'],
                           help="Flag to use FastText embedding")
    argparser.add_argument("-ckpt", type=str, help="model checkpoint file to restart training")


    # ========= NLP Model parameters
    argparser.add_argument("-encoder", type=str, default='ff', choices=['sru', 'lstm', 'cnn'], help="which model class to use")
    argparser.add_argument("-similarity_measure", type=str, default='dot_product', choices=['cosine',  'dot_product', 'euclidean'], 
                            help="which model class to use")

    argparser.add_argument("-word_vec_size", default=300, type=int, help="size of wordvector")
    argparser.add_argument("-n_d", type=int, default=300, help="hidden dimension")
    argparser.add_argument("-num_layers", "-depth", default=1, type=int, help="number of non-linear layers")
    argparser.add_argument("-activation", type=str, default='tanh', help="activation func")
    argparser.add_argument("-dropout", type=float, default=0.2, help="dropout prob")
    argparser.add_argument("-train_emb", action='store_true', default=False, help="Train the embedding lookup weight")
    argparser.add_argument("-sharing_encoder", action='store_true', default=True, help="Train the embedding lookup weight")
    '''
    argparser.add_argument("-num_lstm", type=int, default=2, help="number of stacking lstm layers")
    argparser.add_argument("-bidirectional", "-bidir", default=True, action="store_true",
                           help="use bi-directional LSTM")
    '''

    # ========= Q learning parameters
    argparser.add_argument("-n_replay", type=int, default=100)
    argparser.add_argument("-update_frequency", type=int, default=200, help='how many rollouts for each update')
    argparser.add_argument("-target_update", type=float, default=10, help='how often to update target q net')
    argparser.add_argument("-learn_start", type=int, default=300, help='start learning when memory size reach certain size')
    argparser.add_argument("-memorysize", type=int, default=10000, help='capacity of memory buffer')


    argparser.add_argument("-gamma", type=float, default=0.999)
    argparser.add_argument("-eps_start", type=float, default=0.98)
    argparser.add_argument("-eps_end", type=float, default=0.05)
    argparser.add_argument("-eps_decay", type=int, default=200)
    argparser.add_argument("-reward_shaping", action='store_true', default=False, help="Using faq vector to provide reward")
    argparser.add_argument("-turnpenalty", type=float, default=0.05)
    argparser.add_argument("-debug", action='store_true', default=False, help=" ")


    # ========= Optimization parameters
    argparser.add_argument("-batch_size", type=int, default=64)
    argparser.add_argument("-eval_batch_size", type=int, default=1200, help='Batch size during evaluation')
    argparser.add_argument("-max_epoch", type=int, default=100)


    argparser.add_argument("-learning", type=str, default='adam')
    argparser.add_argument("-warmup_steps", type=int, default=4000, help="Learning rate warmup steps")
    argparser.add_argument("-lr", type=float, default=0.001, help="Initial learning rate")
    argparser.add_argument("-lr_gamma", type=float, default=0.5, help="Proportion by which to cut lr")
    argparser.add_argument("-lr_step", type=int, default=20, help="Number of epochs between lr decrease")
    argparser.add_argument("-patience", type=int, default=5,
                           help="Number of epochs of non-increasing val auc0.05 before cutting lr in half")
    argparser.add_argument("-lr_scheduler", type=str, default='step', choices=['noam', 'step', 'val'],
                           help='Learning rate scheduler.'
                                '"noam" is linear growth until warmup_steps and then 1/x decay.'
                                '"step" reduces the learning rate by lr_gamma every lr_step epochs.'
                                '"val" cuts the learning rate by lr_gamma every time the val auc0.05 doesn\'t'
                                'increase for patience epochs.')
    argparser.add_argument("-max_grad_norm", default=3.5, type=float, help='Maximum gradient norm')
    argparser.add_argument("-adam_beta1", type=float, default=0.9,
                           help='The beta1 parameter used by Adam. adam_default 0.9')
    argparser.add_argument("-adam_beta2", type=float, default=0.98,
                           help='The beta1 parameter used by Adam. adam_default 0.999')
    argparser.add_argument("-adam_eps", type=float, default=1e-9,
                           help='The beta1 parameter used by Adam. adam_default 1e-8')
    argparser.add_argument("-weight_decay", type=float, default=0.0, help='L2 regulation, adam_default 0.0')



    args = argparser.parse_args()
    main(args)
