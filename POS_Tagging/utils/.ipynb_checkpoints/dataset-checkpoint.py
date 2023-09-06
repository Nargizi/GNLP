from dataclasses import dataclass, field
from typing import TextIO, Protocol
import json

import pandas as pd
import os

import torchtext

from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

UNK_TOKEN = '<unk>'


class POSDataSet(Dataset):
    def __init__(self, data_frame: pd.DataFrame, x_col: str = 'mod_words', y_col: str = 'pos_tags'):
        self.data = data_frame
        self.x_col = x_col
        self.y_col = y_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (str, str):
        return self.x.iloc[idx], self.y.iloc[idx]

    @property
    def x(self):
        return self.data[self.x_col]

    @property
    def y(self):
        return self.data[self.y_col]

class VocabCreator(Protocol):

    def make(self) -> torchtext.vocab.Vocab:
        pass
    
    
class POSFolderDataSet(Dataset):
    def __init__(self, folder_path, start_with: str = "sent"):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith(".orc") and f.startswith(start_with)]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        df = pd.read_orc(file_path, dtype_backend='pyarrow')
        
        return {"data": df}


class VocabCreator(Protocol):

    def make(self) -> torchtext.vocab.Vocab:
        pass
    

@dataclass
class DataSetVocabCreator:
    dataset: POSDataSet
    unk_token: str = field(default=UNK_TOKEN)

    @staticmethod
    def __get_tokens(word_iter):
        yield from list(word_iter)

    def make(self) -> torchtext.vocab.Vocab:
        vocab = build_vocab_from_iterator(iterator=self.__get_tokens(self.dataset.x),
                                          specials=[self.unk_token],
                                          special_first=False
                                          )
        vocab.set_default_index(vocab[self.unk_token])

        return vocab

@dataclass
class FileVocabCreator:
    source: TextIO
    unk_token: str = field(default=UNK_TOKEN)

    @staticmethod
    def __get_tokens(word_iter):
        for word in word_iter:
            yield str(word)

    def make(self) -> torchtext.vocab.Vocab:
        vocab = build_vocab_from_iterator(iterator=[self.__get_tokens(json.load(self.source))],
                                          specials=[self.unk_token],
                                          special_first=False
                                          )
        vocab.set_default_index(vocab[self.unk_token])

        return vocab
