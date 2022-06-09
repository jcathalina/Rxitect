"""FIXME"""
import time
import joblib
import numpy as np
import pytorch_lightning as pl
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from rxitect.process_qsar_data import construct_qsar_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiTaskDNN(pl.LightningModule):
    """
    """

    def __init__(self, n_dim, n_task, max_epochs: int = 100) -> None:
        super(MultiTaskDNN, self).__init__()

        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 4_000)
        self.fc1 = nn.Linear(4_000, 2_000)
        self.fc2 = nn.Linear(2_000, 1_000)
        self.output = nn.Linear(1_000, n_task)
        self.loss = nn.MSELoss
        self.max_epochs = max_epochs
        self.to(device)

    def forward(self, X: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        y = F.relu(self.fc0(X))
        if is_training:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if is_training:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if is_training:
            y = self.dropout(y)
        y = self.output(y)
        return y

    def fit(self, train_loader, test_loader, epochs=100, lr=1e-4):
        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # record the minimum loss value based on the calculation of loss function by the current epoch
        best_loss = np.inf
        # record the epoch when optimal model is saved.
        last_save = 0
        for epoch in tqdm(range(epochs), desc="Fitting MT DNN"):
            t0 = time.time()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - 1 / epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(train_loader):
                # Batch of target tenor and label tensor
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                # predicted probability tensor
                y_ = self(Xb, is_training=True)
                # ignore all of the NaN values
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                wb = torch.Tensor(yb.size()).to(device)
                wb[yb == 3.99] = 0.1
                wb[yb != 3.99] = 1
                # loss function calculation based on predicted tensor and label tensor
                loss = self.loss(y_ * wb, yb * wb)
                loss.backward()
                optimizer.step()
            # loss value on validation set based on which optimal model is saved.
            loss_valid = self.evaluate(test_loader)
            print('[Epoch: %d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                epoch, epochs, time.time() - t0, loss.item(), loss_valid))
            if loss_valid < best_loss:
                print('[Performance] loss_valid is improved from %f to %f' %
                      (best_loss, loss_valid))
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid is not improved.')
                # early stopping, if the performance on validation is not improved in 100 epochs.
                # The model training will stop in order to save time.
                if epoch - last_save > 100: break

if __name__ == "__main__":
    targets = ["CHEMBL226", "CHEMBL240", "CHEMBL251"]
    cols = [
            "target_chembl_id",
            "smiles",
            "pchembl_value",
            "comment",
            "standard_type",
            "standard_relation",
            "document_year",
        ]
    px_placeholder = 3.99
    temporal_split_year = 2015

    dataset = construct_qsar_dataset("../data/raw/ligand_raw.tsv",
                                    targets,
                                    cols)
    X_train = dataset.X_train(targets[0])
    y_train = dataset.y_train(targets[0])

    mt_dnn = MultiTaskDNN(n_dim=X_train.shape[1], n_tasks=y_train.shape[1])
    train_loader = DataLoader(dataset.df_train) 
    
    trainer = pl.Trainer()
    trainer.fit(model=mt_dnn, train_dataloaders=train_loader)
