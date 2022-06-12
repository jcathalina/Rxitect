from typing import List
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from sklearn.model_selection import KFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.out_layer(x)
        return x

class LSTMEncoder(pl.LightningModule):
    def __init__(self, input_dim, encoder_dim, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_dim
        self.n_directions = 1
        self.encoder_dim = encoder_dim
        
        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=encoder_dim,
                           num_layers=num_layers,
                           dropout=dropout,
                           batch_first=True)
        
    def forward(self, inp, previous_hidden=None, pack=True):
        input_tensor = inp[0]
        input_length = inp[1]
        print(f"INP LSTM: {inp}")
        # print(f"INP TENSOR: {inp[0]}")
        # print(f"INP LEN: {inp[1]}")
        batch_size = input_tensor.size(0)
        # TODO: warning: output shape is changed! (batch_first=True) Check hidden
        if pack:
            input_lengths_sorted, perm_idx = torch.sort(input_length, dim=0, descending=True)
            input_lengths_sorted = input_lengths_sorted.detach().to(device="cpu").tolist()
            input_tensor = torch.index_select(input_tensor, 0, perm_idx)
            rnn_input = pack_padded_sequence(input=input_tensor,
                                             lengths=input_lengths_sorted,
                                             batch_first=True)
        else:
            rnn_input = input_tensor
            
        if previous_hidden is None:
            previous_hidden = self.init_hidden(batch_size)
            cell = self.init_cell(batch_size)
            previous_hidden = (previous_hidden, cell)
        else:
            hidden = previous_hidden[0]
            cell = previous_hidden[1]
            hidden = torch.index_select(hidden, 1, perm_idx)
            cell = torch.index_select(cell, 1, perm_idx)
            previous_hidden = (hidden, cell)
        rnn_output, next_hidden = self.rnn(rnn_input)

        if pack:
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
            _, unperm_idx = perm_idx.sort(0)
            rnn_output = torch.index_select(rnn_output, 0, unperm_idx)
            hidden = next_hidden[0]
            cell = next_hidden[1]
            hidden = torch.index_select(hidden, 1, unperm_idx)
            cell = torch.index_select(cell, 1, unperm_idx)
            next_hidden = (hidden, cell)

        index_t = (input_length - 1).to(dtype=torch.long)
        index_t = index_t.view(-1, 1, 1).expand(-1, 1, rnn_output.size(2))

        embedded = torch.gather(rnn_output, dim=1, index=index_t).squeeze(1)

        return embedded, next_hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.encoder_dim)

    def init_cell(self, batch_size):
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.encoder_dim)


class Embedding(pl.LightningModule):
    def __init__(self, embedding_dim, n_embeddings, padding_idx=None):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = n_embeddings
        self.padding_idx = padding_idx if padding_idx else None
        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=self.padding_idx)

    def forward(self, inp):
        inp = inp.type(torch.long)
        # print(f"INPUT: {inp}")
        embedded = self.embedding(inp)
        return embedded


class PChEMBLValueRegressor(pl.LightningModule):
    def __init__(self, tokens: List[str], lr: float):
        super(PChEMBLValueRegressor, self).__init__()
        self.tokens = tokens
        self.lr = lr
        self.embedding = Embedding(embedding_dim=128, n_embeddings=len(self.tokens), padding_idx=tokens.index(' '))
        self.encoder = LSTMEncoder(input_dim=128, encoder_dim=128, num_layers=2, dropout=0.8)
        self.mlp = MLP(input_dim=128, hidden_dim=128, output_dim=1)

    def forward(self, inp, eval=False):
        if eval:
            self.eval()
        else:
            self.train()
        input_tensor = inp[0]
        input_length = inp[1]
        embedded = self.embedding(input_tensor)
        # print(f"EMBEDDED: {embedded}")
        # print(f"INPUT LENGTH IN PREDICTOR: {input_length}")
        output, _ = self.encoder([embedded, input_length])
        output = self.mlp(output)
        return output


    # @staticmethod
    # def cast_inputs(sample, task, use_cuda, for_prediction=False):
    #     batch_mols = sample['tokenized_smiles'].to(dtype=torch.long)
    #     if for_prediction and "object" in sample.keys():
    #         batch_object = sample['object']
    #     else:
    #         batch_object = None
    #     batch_length = sample['length'].to(dtype=torch.long)
    #     if not for_prediction and "labels" in sample.keys():
    #         batch_labels = sample['labels'].to(dtype=torch.float)
    #         if task == 'classification':
    #             batch_labels = batch_labels.to(dtype=torch.long)
    #     else:
    #         batch_labels = None
    #     if use_cuda:
    #         batch_mols = batch_mols.to(device="cuda")
    #         batch_length = batch_length.to(device="cuda")
    #         if batch_labels is not None:
    #             batch_labels = batch_labels.to(device="cuda")
    #     if batch_object is not None:
    #         return (batch_mols, batch_length), batch_object
    #     else:
    #         return (batch_mols, batch_length), batch_labels

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # print(f"BATCH: {batch}")
        x, y = batch['tokenized_smiles'], batch['labels']
        y_pred = self(x)
        loss = nn.MSELoss(y_pred, y)
        return loss


def identity(input):
    return input


def get_tokens(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and
    number of unique tokens from the list of SMILES
    Args:
        smiles (list): list of SMILES strings to tokenize.
        tokens (string): string of tokens or None.
        If none will be extracted from dataset.
    Returns:
        tokens (list): list of unique tokens/SMILES alphabet.
        token2idx (dict): dictionary mapping token to its index.
        num_tokens (int): number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = sorted(tokens)
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens