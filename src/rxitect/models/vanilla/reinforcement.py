
from datetime import date, datetime
from typing import List, Optional, Union
import numpy as np
import torch
import selfies as sf
import pandas as pd
from tqdm import trange, tqdm
from globals import device
from rxitect import tensor_utils
from rxitect.models.lightning.generator import Generator
from rxitect.structs.environment import Environment, ScoringScheme
from rxitect.structs.vocabulary import SelfiesVocabulary
from globals import root_path
from torch.utils.data import TensorDataset, DataLoader
from rxitect.models.vanilla.generator import VanillaGenerator
import matplotlib.pyplot as plt
import mlflow.pytorch
import mlflow


class Rxitect:
    def __init__(self, generator: Union[Generator, VanillaGenerator],
                 environment: Environment,
                 replay_size: int = 10,
                 batch_size: int = 64,
                 sample_size: int = 128,):
        self.generator = generator
        self.env = environment

        self.replay: int = replay_size
        self.batch_size: int = batch_size
        self.n_samples: int = sample_size
        # self.mean_fn: str = "geometric"

    def policy_gradient(self,
                        crossover_net: Optional[VanillaGenerator] = None,
                        mutation_net: Optional[VanillaGenerator] = None) -> None:
        # print(">>> Policy Gradient step...")
        sequences = [evolve(net=self.generator,
                            batch_size=self.batch_size,
                            crover=crossover_net,
                            mutate=mutation_net)
                            for _ in trange(self.replay, desc="Generating sequences")]
        sequences = torch.cat(tensors=sequences, dim=0)
        selfies = np.array([self.generator.voc.decode(seq) for seq in tqdm(sequences, desc="Decoding to SELFIES")])
        uidx = tensor_utils.unique(np.array([[selfie] for selfie in selfies]))
        selfies = selfies[uidx]
        smiles = [sf.decoder(selfie) for selfie in list(selfies)]
        sequences = sequences[torch.tensor(uidx, device=device, dtype=torch.long)]
        
        # print(">>> Scoring...")
        scores = self.env.calc_reward(smiles=smiles, scheme="PR")
        dataset = TensorDataset(sequences, torch.tensor(scores, device=device))
        dataloader = DataLoader(dataset=dataset, batch_size=self.n_samples, shuffle=True)

        self.generator.policy_grad_loss(loader=dataloader)

    def fit(self,
            max_epochs: int = 1_000,
            use_crossover: bool = False,
            use_mutation: bool = False, 
            use_mflow: bool = False):

        time_tag = datetime.strftime(date.today(), format="%d%m%Y")
        log_filepath = root_path / f"logs/rxitect_run_{time_tag}.log"
        best_score = 0
        last_save = 0

        crossover_net = _init_crossover_net() if use_crossover else None
        mutation_net = _init_mutation_net() if use_mutation else None

        if use_mflow:
            mlflow.log_params({
                "epochs": max_epochs,
                "crossover": use_crossover,
                "mutation": use_mutation,
                "lr": self.generator.lr,
                "scoring_scheme": self.env.scoring_scheme.name
            })

        with open(log_filepath, "w") as log_file:
            print("\t".join([key for key in self.env.keys] + ["\tDESIRE"]), file=log_file)
            for epoch in trange(max_epochs, desc="RL Loop"):

                self.policy_gradient(crossover_net=crossover_net,
                                     mutation_net=mutation_net)

                sequences = self.generator.sample(self.n_samples)
                uidx = tensor_utils.unique(sequences)
                selfies = [self.generator.voc.decode(seq) for seq in sequences[uidx]]
                smiles = [sf.decoder(x) for x in selfies]
                scores = self.env.get_preds(mols=smiles, is_smiles=True)
                
                desire = scores["DESIRE"].sum() / self.n_samples
                score = scores[self.env.keys].values.mean()

                if use_mflow:
                    dyn_metrics = {k: scores[k].mean() for k in self.env.keys}
                    metrics = {
                        "score": score,
                        "desire": desire,
                        **dyn_metrics,
                    }
                    mlflow.log_metrics(metrics=metrics, step=epoch)

                if epoch %25 == 0 or epoch == max_epochs-1:
                    print(f"SMILES @ EPOCH {epoch}: {'\n'.join(smiles)}", file=log_file)

                if best_score <= score and (epoch >= 25):
                    torch.save(self.generator.state_dict(), root_path / f"models/{time_tag}_epoch={epoch}.pkg")
                    best_score = score
                    last_save = epoch
                
                # print(f"Epoch: {epoch} average: {score:.4} frac desirable: {desire}", file=log_file)
                
                # for i, smile in enumerate(smiles):
                #     score_log = "\t".join([f"{sc:.3}" for sc in scores.values[i]])
                #     print(f"{score_log}\t{smile}", file=log_file)

                if epoch - last_save > 100:  # early stop criterium
                    break
            
            for param_group in self.generator.optim.param_groups:
                param_group["lr"] *= 0.99


def _init_crossover_net() -> VanillaGenerator:
    voc = SelfiesVocabulary(vocabulary_file_path=root_path / "data/processed/selfies_voc.txt")
    crossover_net = VanillaGenerator(voc=voc)
    crossover_net.load_state_dict(torch.load(root_path / "models/fine_tuned_selfies_rnn.ckpt")["state_dict"])
    return crossover_net.eval()


def _init_mutation_net() -> VanillaGenerator:
    voc = SelfiesVocabulary(vocabulary_file_path=root_path / "data/processed/selfies_voc.txt")
    mutation_net = VanillaGenerator(voc=voc)
    mutation_net.load_state_dict(torch.load(root_path / "models/pretrained_selfies_rnn.ckpt")["state_dict"])
    return mutation_net.eval()


def evolve(net: Union[Generator, VanillaGenerator],
           batch_size: int,
           epsilon: float = 1e-3,
           crover: Optional[VanillaGenerator] = None,
           mutate: Optional[VanillaGenerator] = None,) -> torch.Tensor:
    """
    """
    # Start tokens
    x = torch.tensor(
        [net.voc.tk2ix["GO"]] * batch_size, dtype=torch.long, device=device
    )
    # Hidden states initialization for exploitation network
    h = net.init_h(batch_size)
    # Hidden states initialization for exploration network
    h2 = net.init_h(batch_size)
    # Initialization of output matrix
    sequences = torch.zeros(batch_size, net.voc.max_len, device=device).long()
    # labels to judge and record which sample is ended
    is_end = torch.zeros(batch_size, device=device).bool()

    for step in range(net.voc.max_len):
        is_change = torch.rand(1) < 0.5
        if crover is not None and is_change:
            logit, h = crover(x, h)
        else:
            logit, h = net(x, h)
        proba = logit.softmax(dim=-1)
        if mutate is not None:
            logit2, h2 = mutate(x, h2)
            ratio = torch.rand(batch_size, 1, device=device) * epsilon
            proba = (
                logit.softmax(dim=-1) * (1 - ratio) + logit2.softmax(dim=-1) * ratio
            )
        # sampling based on output probability distribution
        x = torch.multinomial(proba, 1).view(-1)

        x[is_end] = net.voc.tk2ix["EOS"]
        sequences[:, step] = x

        # Judging whether samples are end or not.
        end_token = x == net.voc.tk2ix["EOS"]
        is_end = torch.ge(is_end + end_token, 1)
        #  If all of the samples generation being end, stop the sampling process
        if (is_end == 1).all():
            break
    return sequences


if __name__ == "__main__":
    
    print("Initiating ML Flow tracking...")
    mlflow.set_tracking_uri("https://dagshub.com/naisuu/Rxitect.mlflow")
    with mlflow.start_run() as run:
        voc = SelfiesVocabulary(root_path / "data/processed/selfies_voc.txt")
        gen = VanillaGenerator(voc=voc)
        gen.load_state_dict(torch.load(root_path / "models/fine_tuned_selfies_rnn.ckpt")["state_dict"])
        env = Environment.get_default_env(scoring_scheme=ScoringScheme.PR)
        rxitect = Rxitect(generator=gen, environment=env)
        rxitect.fit(max_epochs=100,
                    use_crossover=True,
                    use_mutation=True,
                    use_mflow=True)


    # x = evolve(net=gen, batch_size=64)
    # x_dec = [voc.decode(y) for y in x]
    # smi_x = [sf.decoder(y) for y in x_dec]
    # res = voc.check_smiles(sequences=x)
    # print(f"frac val: {np.sum(res[1]) / len(res[1])}")
