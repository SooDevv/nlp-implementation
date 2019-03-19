import pandas as pd
import torch
from torch.utils.data import Dataset
from konlpy.tag import Okt
from gluonnlp.data import PadSequence
from gluonnlp import Vocab


class Corpus(Dataset):
    def __init__(self, data, vocab: Vocab, tagger: Okt, padder: PadSequence) -> None:
        self.corpus = data
        self.vocab = vocab
        self.tagger = tagger
        self.padder = padder

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        tokenized = self.tagger.morphs(self.corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self.padder([self.vocab.token_to_idx[token] for token in tokenized]))
        labels = torch.tensor(self.corpus.iloc[idx]['label'])
        return tokenized2indices, labels
