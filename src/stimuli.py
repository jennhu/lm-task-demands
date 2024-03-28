import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset

class Dataset():
    """Base class for evaluation datasets."""
    def __init__(self, name: str):
        self.name = name
        self.items = self.load_items()

    def load_items(self):
        raise NotImplementedError
    
class CSVDataset(Dataset):
    """Class for datasets based on a CSV file in the `external_stimuli` folder."""
    def load_items(self, stimuli_folder: str = "external_stimuli"):
        df = pd.read_csv(
            # File name format: external_stimuli/{task}/stimuli.csv
            Path(stimuli_folder, self.name, "stimuli.csv")
        )
        self.n_items = len(df)
        return df

class DigitMatDataset(Dataset):
    """Bespoke class for the Digit Matrices reasoning task (Webb et al. 2023)."""
    def load_items(self):
        problems = np.load(
            Path("external_stimuli", "digit_mat", "all_problems.npz"), 
            allow_pickle=True
        )
        self.n_items = len(problems)
        return problems

class LambadaDataset(Dataset):
    """Bespoke class for the Lambada word prediction task (Paperno et al. 2016)."""
    def load_items(self):
        dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        self.n_items = dataset.num_rows
        ids = dataset.to_iterable_dataset()
        return ids
    
TASK_TO_DATASET = {
    "lambada": LambadaDataset,
    "digit_mat": DigitMatDataset,
    "crt": CSVDataset,
    "dgl": CSVDataset,
    "blimp": CSVDataset
}