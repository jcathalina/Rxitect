from pyprojroot import here
import torch

root_path = here(project_files=[".here"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")