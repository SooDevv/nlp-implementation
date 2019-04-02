import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp.data import PadSequence
from utils import JamoTokenizer



class Corpus(Dataset):
    def __init__(self, data, tokenizer: JamoTokenizer, padder: PadSequence) -> None:
        self._corpus = data
        self._padder = padder
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        tokenized2indices = self._tokenizer.tokenize_and_transform(self._corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self._padder(tokenized2indices))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokenized2indices, label
