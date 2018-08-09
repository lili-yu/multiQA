"""
encoder.py
Partly From mytorch
"""

import torch
import torch.nn as nn

#from .modules import CNN, RNN, FullyConnected

class Encoder(nn.Module):

    def __init__(self, embedding_layer, args):
        super(Encoder, self).__init__()

        self.n_e = embedding_layer.n_d  # embedding_size
        self.n_d = args.n_d   # hidden dimension 


        #  ============ Embedding ============ 
        if args.embed_coding == 'seq':
            self.embedding = embedding_layer.embedding
        elif args.embed_coding == 'avg':
            embedding = embedding_layer.embedding
            self.embedding = lambda x: embedding(x).mean(1)
        elif args.embed_coding == 'id':
            embedding_module = lambda x: x
        else:
            raise ValueError("Invalid embedding module: {}".format(args.embed_coding))


        # ============ Dropout ============ 
        self.dropout = args.dropout
        self.dropout_op = nn.Dropout(self.dropout)

        # ============ Encoding  ============ 
        self.encoding = args.encoder
        if self.encoding in ('sru', 'lstm', 'gru'):
            self.num_lstm = args.num_lstm
            self.num_layers = args.num_layers
            self.bidirectional = args.bidirectional
            self.rnn_dropout = args.rnn_dropout
            self.atten_reg_coef = args.atten_reg_coef
            self.module = RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              rnn_type=encoder_module,
                              dropout=dropout, **kwargs)

        elif self.encoding == 'cnn':
            inputsize =   embedding_size * args.cnn_max_len 
            self.module = CNN(input_size=inputsize,
                              hidden_size=hidden_size, **kwargs)

        elif self.encoding == 'ff':
            self.num_layers = 2
            seq = nn.Sequential(
                nn.Linear(self.n_e, self.n_d),
                nn.Tanh(),
                self.dropout_op,
                nn.Linear(self.n_d, self.n_d),
                nn.Tanh(),
                self.dropout_op
                )
            self.module = seq

        else:
            raise ValueError("Invalid encoder module: {}".format(encoder_module))

    def forward(self, data, state=None):
        """Performs a forward pass through the network.
        Parameters
        ----------
        data: torch.Variable or tuple
            input to the model of shape (seq_len, batch_size), or
            tuple with the input data and the lenght of each sequence
            in the mini batch of shape (batch_size)
        Returns
        -------
        output: torch.Variable
            if self.reduce is False, has shape (seq_len, batch_size, hidden_size)
            if self.reduce is True, has shape (batch_size, hidden_size)
        state: torch.Variable (for RNN only)
            states of the model of shape (n_layers, batch_size, hidden_size)
        """
        
        embedded = self.embedding(data)
        embedded = self.dropout_op(embedded)
        output = self.module(embedded)

        return output

    def reset_parameters(self):
        """Reset model parameters."""
        if hasattr(self.embedding, 'weight') and self.embedding.weight.requires_grad:
            self.embedding.reset_parameters()
        if hasattr(self.module, 'parameters')  and self.module.parameters():
            self.module.reset_parameters()

