import numpy as np
import torch
import torch.nn as nn
from torch import optim

from rxitect.classic.utils import tensor_utils
from rxitect.classic.utils.vocabulary import SelfiesVocabulary


class GeneratorRNN(nn.Module):
    def __init__(
        self,
        voc: SelfiesVocabulary,
        embed_size: int = 128,
        hidden_size: int = 512,
        is_lstm: bool = True,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        super().__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, input, h):
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size, labels=None):
        h = torch.rand(3, batch_size, 512, device=self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size, device=self.device)
        return (h, c) if self.is_lstm else h

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        x = torch.tensor([self.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=self.device)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len, device=self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            logits = logits.log_softmax(dim=-1)
            score = logits.gather(1, target[:, step : step + 1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def policy_grad_loss(self, loader):
        for seq, reward in loader:
            self.zero_grad()
            score = self.likelihood(seq)
            loss = score * reward
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

    def sample(self, batch_size):
        x = torch.tensor([self.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=self.device)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len, dtype=torch.long, device=self.device)
        isEnd = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix["EOS"]
            sequences[:, step] = x

            end_token = x == self.voc.tk2ix["EOS"]
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all():
                break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + ".log", "w")
        best_error = np.inf
        for epoch in range(epochs):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(self.device))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                if i % 10 == 0 or loader_valid is not None:
                    seqs = self.sample(len(batch * 2))
                    ix = tensor_utils.unique(seqs)
                    seqs = seqs[ix]
                    smiles, valids = self.voc.check_smiles(seqs)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (
                        epoch,
                        i,
                        error,
                        loss_train.item(),
                    )
                    if loader_valid is not None:
                        loss_valid, size = 0, 0
                        for j, vbatch in enumerate(loader_valid):
                            size += vbatch.size(0)
                            loss_valid += (
                                -self.likelihood(vbatch.to(self.device)).sum().item()
                            )
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid < best_error:
                            torch.save(self.state_dict(), out + ".pkg")
                            best_error = loss_valid
                        info += " loss_valid: %.3f" % loss_valid
                    elif error < best_error:
                        torch.save(self.state_dict(), out + ".pkg")
                        best_error = error
                    print(info, file=log)
                    for k, smile in enumerate(smiles):
                        print("%d\t%s" % (valids[k], smile), file=log)
        log.close()
