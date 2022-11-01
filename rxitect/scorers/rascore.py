# A PyTorch implementation of RAScoreNN by reymond-group @ https://github.com/reymond-group/RAscore using the
# parameters they found to work best through hyper-optimization. Thakkar,  A.; Chadimová, V.; Bjerrum,
# E. J.; Engkvist, O.; Reymond, J.-L. Retrosynthetic Accessibility Score (RAscore) – Rapid Machine Learned
# Synthesizability Classification from AI Driven Retrosynthetic Planning. Chem. Sci. 2021.
# https://doi.org/10.1039/d0sc05401a

import os
import shutil
import time
from typing import Tuple, Union, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from pyprojroot import here
from rdkit.Chem import AllChem as Chem, Mol
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

tqdm.pandas()


class AiZynthFinderClassificationDataset(Dataset):
    def __init__(self, train_filepath: os.PathLike, fast_dev_run: bool = False) -> None:
        train_data = pd.read_csv(train_filepath, nrows=500 if fast_dev_run else None)
        train_data["descriptor"] = train_data["smi"].progress_apply(_compute_descriptors)
        self.train_X = np.stack(train_data["descriptor"].values).astype(np.float32)
        self.train_y = np.stack(train_data["activity"].values).astype(np.float32).reshape(-1, 1)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.train_X[index], self.train_y[index]

    def __len__(self) -> int:
        return len(self.train_X)


def _compute_descriptors(smiles: str, radius: int = 3, use_features: bool = True,
                         use_counts: bool = True) -> NDArray:
    mol = Chem.MolFromSmiles(smiles)
    fp = Chem.GetMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features)
    size = 2_048
    arr = np.zeros((size,), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        arr[nidx] += int(v)
    return arr


class RAScoreNet(nn.Module):
    def __init__(self, num_features: int = 2_048) -> None:
        super().__init__()
        self.input_layer = nn.Linear(num_features, 512)
        self.lin0 = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.45834579304621176),
            nn.Linear(512, 128), nn.Dropout(0.20214636121010582),
            nn.Linear(128, 512), nn.ELU(), nn.Dropout(0.13847113009081813),
            nn.Linear(512, 256), nn.Dropout(0.21312873496871235),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.33530504087548707),
            nn.Linear(128, 128), nn.Dropout(0.11559123444807062),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2618908919792556),
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.3587291059530903),
            nn.Linear(512, 512), nn.SELU(), nn.Dropout(0.43377277017943133)
        )
        self.output_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_layer(x)
        x = self.lin0(x)
        out = self.output_layer(x)
        return out


def mol2fcfp(mol: Mol, radius: int = 3) -> NDArray:
    fp = Chem.GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=True)
    size = 2_048
    arr = np.zeros((size,), np.int32)
    for idx, v in fp.GetNonzeroElements().items():
        nidx = idx % size
        arr[nidx] += int(v)
    return arr


def load_rascore_model(ckpt_filepath: os.PathLike, device: str = "cpu"):
    net = RAScoreNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1.5691774834712003e-05)

    # load checkpoint if needed/ wanted
    ckpt = load_checkpoint(ckpt_dir_or_file=ckpt_filepath)  # custom method for loading last checkpoint
    net.load_state_dict(ckpt['net'])
    optim.load_state_dict(ckpt['optim'])
    print("Checkpoint restored!")
    net.eval()

    return net
#
# class RAScore:
#     def __init__(self, ckpt_filepath: os.PathLike, device: str = "cpu") -> None:
#         self.net = RAScoreNet().to(device)
#         optim = torch.optim.Adam(self.net.parameters(), lr=1.5691774834712003e-05)
#
#         # load checkpoint if needed/ wanted
#         ckpt = load_checkpoint(ckpt_dir_or_file=ckpt_filepath)  # custom method for loading last checkpoint
#         self.net.load_state_dict(ckpt['net'])
#         optim.load_state_dict(ckpt['optim'])
#         print("Checkpoint restored!")
#         self.net.eval()
#
#     def score_single_smiles(self, smiles: str) -> float:
#         try:
#             arr = _compute_descriptors(smiles).astype(np.float32)
#         except ValueError:
#             print("SMILES could not be converted to ECFP6 count vector")
#             return float("NaN")
#
#         arr = torch.from_numpy(arr)
#         proba = self.net(arr.reshape(1, -1))
#         return proba[0][0]


@hydra.main(version_base=None, config_path="config", config_name="ra_score")
def train_ra_score_dnn(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(12345)
    device = cfg.device
    dataset = AiZynthFinderClassificationDataset(
        train_filepath=here() / "data/uspto_chembl_classification_train.csv", fast_dev_run=cfg.fast_dev_run)
    test_dataset = AiZynthFinderClassificationDataset(
        train_filepath=here() / "data/uspto_chembl_classification_test.csv", fast_dev_run=cfg.fast_dev_run)
    train_data_loader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=(device != "cpu"), shuffle=True,
                                   num_workers=cfg.num_workers)
    test_data_loader = DataLoader(test_dataset, pin_memory=(device != "cpu"), shuffle=False,
                                  num_workers=cfg.num_workers)
    net = RAScoreNet().to(device)
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr=1.5691774834712003e-05)

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if cfg.resume and cfg.path_to_checkpoint is not None:
        ckpt = load_checkpoint(cfg.path_to_checkpoint)  # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("Checkpoint restored!")

    writer = SummaryWriter()

    n_iter = start_n_iter
    best_test_loss = np.inf
    for epoch in range(start_epoch, cfg.epochs):
        # set models to train mode
        net.train()

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_data_loader),
                    total=len(train_data_loader))
        start_time = time.time()

        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            fingerprints, label = data
            fingerprints = fingerprints.to(device)
            label = label.to(device)

            # It's very good practice to keep track of preparation time and computation time using tqdm to find any
            # issues in your dataloader
            prepare_time = start_time - time.time()

            # forward and backward pass
            out = net(fingerprints)
            loss = F.binary_cross_entropy(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # udpate tensorboard
            writer.add_scalar('train_loss', loss.item(), n_iter)

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {loss.item():.2f},  epoch: {epoch}/{cfg.epochs}')
            start_time = time.time()

        # TODO: maybe do a test pass every N=1 epochs
        if not cfg.fast_dev_run and epoch % cfg.validate_every_n_epochs == 0:
            # bring models to evaluation mode
            net.eval()
            test_loss = 0.0
            pbar = tqdm(enumerate(test_data_loader),
                        total=len(test_data_loader))
            with torch.no_grad():
                for i, data in pbar:
                    # data preparation
                    fingerprints, label = data
                    fingerprints = fingerprints.to(device)
                    label = label.to(device)

                    out = net(fingerprints)

                    test_loss = F.binary_cross_entropy(out, label)

            print(f'Loss on test set: {test_loss.item()}')
            writer.add_scalar('test_loss', test_loss.item(), n_iter)

            if test_loss.item() < best_test_loss:
                print("model accuracy improved!")
                # save checkpoint if needed
                cpkt = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'n_iter': n_iter,
                    'optim': optim.state_dict()
                }
                save_checkpoint(cpkt, 'model_checkpoint.ckpt')


def evaluate_ra_score_on_test_set(test_filepath: os.PathLike, ckpt_filepath: os.PathLike, device: str = "cpu") -> None:
    torch.manual_seed(12345)
    test_dataset = AiZynthFinderClassificationDataset(
        train_filepath=test_filepath, fast_dev_run=False)
    test_data_loader = DataLoader(test_dataset, pin_memory=(device != "cpu"), shuffle=False, num_workers=4)
    net = RAScoreNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1.5691774834712003e-05)

    # load checkpoint if needed/ wanted
    ckpt = load_checkpoint(ckpt_dir_or_file=ckpt_filepath)  # custom method for loading last checkpoint
    net.load_state_dict(ckpt['net'])
    optim.load_state_dict(ckpt['optim'])
    print("Checkpoint restored!")

    # bring models to evaluation mode
    net.eval()
    test_loss = 0.0
    pbar = tqdm(enumerate(test_data_loader),
                total=len(test_data_loader))
    with torch.no_grad():
        for i, data in pbar:
            # data preparation
            fingerprints, label = data
            fingerprints = fingerprints.to(device)
            label = label.to(device)

            out = net(fingerprints)

            test_loss = F.binary_cross_entropy(out, label)

    print(f'Loss on test set: {test_loss.item()}')


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: Union[os.PathLike, str], map_location: Optional[Union[str, torch.device]] = None, load_best: bool = False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt
