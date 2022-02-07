import os
import time

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import yaml
from pyprojroot import here
from rdkit.Chem import MolFromSmiles

from rxitect import featurization, utils
from rxitect.models import vae


def train_model(
    vae_encoder,
    vae_decoder,
    data_train,
    data_valid,
    num_epochs,
    batch_size,
    lr_enc,
    lr_dec,
    KLD_alpha,
    sample_num,
    sample_len,
    alphabet,
    type_of_encoding,
):
    """
    Train the Variational Auto-Encoder
    """

    print("num_epochs: ", num_epochs)

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach().to(device)
    num_batches_train = int(len(data_train) / batch_size)

    quality_valid_list = [0, 0, 0, 0]
    for epoch in range(num_epochs):

        data_train = data_train[torch.randperm(data_train.size()[0])]

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx:stop_idx]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            # initialization hidden internal state of RNN (RNN has two inputs
            # and two outputs:)
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # compute ELBO
            loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(
                    vae_encoder, vae_decoder, data_valid, batch_size
                )

                report = (
                    "Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| "
                    "quality: %.4f | quality_valid: %.4f)\t"
                    "ELAPSED TIME: %.5f"
                    % (
                        epoch,
                        batch_iteration,
                        num_batches_train,
                        loss.item(),
                        quality_train,
                        quality_valid,
                        end - start,
                    )
                )
                print(report)
                start = time.time()

        quality_valid = quality_in_valid_set(
            vae_encoder, vae_decoder, data_valid, batch_size
        )
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.0:
            corr, unique = latent_space_quality(
                vae_encoder,
                vae_decoder,
                type_of_encoding,
                alphabet,
                sample_num,
                sample_len,
            )
        else:
            corr, unique = -1.0, -1.0

        report = (
            "Validity: %.5f %% | Diversity: %.5f %% | "
            "Reconstruction: %.5f %%"
            % (corr * 100.0 / sample_num, unique * 100.0 / sample_num, quality_valid)
        )
        print(report)

        with open("results.dat", "a") as content:
            content.write(report + "\n")

        if quality_valid_list[-1] < 70.0 and epoch > 200:
            break

        if quality_increase > 20:
            print("Early stopping criteria")
            break


def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1.0 + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld


def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

    differences = 1.0 - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0.0, max=1.0).double()
    quality = 100.0 * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality


def latent_space_quality(
    vae_encoder, vae_decoder, type_of_encoding, alphabet, sample_num, sample_len
):
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality:" f" Take {sample_num} samples from the latent space")

    for _ in range(1, sample_num + 1):

        molecule_pre = ""
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(" ", "")

        if type_of_encoding == 1:  # if SELFIES, decode to SMILES
            molecule = sf.decoder(molecule)

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)


def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []

    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension, device=device)
    hidden = vae_decoder.init_hidden()

    # runs over letters from molecules (len=size of largest molecule)
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)

        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)

        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms


def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size):
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    for batch_iteration in range(min(25, num_batches_valid)):

        # get batch
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx:stop_idx]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=device)
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # assess reconstruction quality
        quality = compute_recon_quality(batch, out_one_hot)
        quality_list.append(quality)

    return np.mean(quality_list).item()


def main():
    content = open("logfile.dat", "w")
    content.close()
    content = open("results.dat", "w")
    content.close()

    if os.path.exists(root / "config/vae_settings.yml"):
        settings = yaml.safe_load(open(root / "config/vae_settings.yml", "r"))
        print(f"type of settings: {type(settings)}")
    else:
        raise FileNotFoundError("Expected a settings file but didn't find it.")

    print("--> Acquiring data...")
    type_of_encoding = settings["data"]["type_of_encoding"]
    file_name = settings["data"]["smiles_file"]

    print("Finished acquiring data.")

    print("Representation: SELFIES")
    (
        encoding_list,
        encoding_alphabet,
        largest_molecule_len,
    ) = featurization.generate_selfies_encodings(file_path=file_name)

    print("--> Creating one-hot encoding...")
    data = utils.selfies_to_onehot(
        selfies_list=encoding_list, longest_selfies_len=largest_molecule_len, selfies_vocab=encoding_alphabet
    )
    print("Finished creating one-hot encoding.")

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(" ")
    print(
        f"Alphabet has {len_alphabet} letters, "
        f"largest molecule is {len_max_molec} letters."
    )

    data_parameters = settings["data"]
    batch_size = data_parameters["batch_size"]

    encoder_parameter = settings["encoder"]
    decoder_parameter = settings["decoder"]
    training_parameters = settings["training"]

    vae_encoder = vae.Encoder(in_dimension=len_max_mol_one_hot, **encoder_parameter).to(
        device
    )
    vae_decoder = vae.Decoder(
        **decoder_parameter, out_dimension=len(encoding_alphabet)
    ).to(device)

    print("*" * 15, ": -->", device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.5, 0.5, 0.0]
    data = data[torch.randperm(data.size()[0])]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    print("start training")
    train_model(
        **training_parameters,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        batch_size=batch_size,
        data_train=data_train,
        data_valid=data_valid,
        alphabet=encoding_alphabet,
        type_of_encoding=type_of_encoding,
        sample_len=len_max_molec,
    )

    with open("COMPLETED", "w") as content:
        content.write("exit code: 0")


if __name__ == "__main__":
    root = here(project_files=[".here"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
