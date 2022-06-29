
import torch
from pytorch_lightning import LightningModule
from rxitect.structs.vocabulary import Vocabulary
import torch.nn.functional as F

class Sampler:
    def __init__(self, model: LightningModule, vocabulary: Vocabulary) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.max_len = vocabulary.max_len

    def sample(self):
        self.model.eval()
        sample_tensor = torch.zeros((self.max_len, 1), dtype=torch.long, device=self.model.device)
        sample_tensor[0, 0] = self.vocabulary.tk2ix[self.vocabulary.start]
        with torch.no_grad():
            for i in range(self.max_len-1):
                tensor = sample_tensor[:i+1]
                logits = self.model.forward(tensor)[-1]
                probabilities = F.softmax(input=logits, dim=1).squeeze()
                token = torch.multinomial(input=probabilities, num_samples=1)
                sample_tensor[i+1, 0] = token
                if token == self.vocabulary.tk2ix[self.vocabulary.end]:
                    break

            mol_string = "".join(self.vocabulary.decode(sample_tensor.squeeze().detach().cpu().numpy())).strip("".join([self.vocabulary.start, self.vocabulary.end]))
            return mol_string
