import os
import itertools
from typing import List
from dataclasses import dataclass, field

import pandas as pd

import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

from utils.constants import PAD_TOKEN, UNK_TOKEN, EOW_TOKEN


def word_to_seq(word: str, vocab: torchtext.vocab.Vocab) -> torch.Tensor:
    return torch.tensor(vocab(list(word)) + [vocab[EOW_TOKEN]])


def tensors_to_words(t: torch.Tensor,
                     vocab: torchtext.vocab.Vocab) -> List[str]:
    return ["".join([vocab.get_itos()[char] if char != vocab[UNK_TOKEN] else '@' for char in word if
                     char != vocab[PAD_TOKEN] and char != vocab[EOW_TOKEN]]) for word in t]


def coallate_words(batch, max_length: int, vocab: torchtext.vocab.Vocab):
    word_list, lemma_list = [torch.zeros(max_length, dtype=torch.int64)], [torch.zeros(max_length, dtype=torch.int64)]
    for word, lemma in batch:
        word_list.append(word_to_seq(word, vocab))
        lemma_list.append(word_to_seq(lemma, vocab))

    return pad_sequence(word_list, batch_first=True, padding_value=vocab[PAD_TOKEN])[1:], pad_sequence(lemma_list,
                                                                                                       batch_first=True,
                                                                                                       padding_value=
                                                                                                       vocab[
                                                                                                           PAD_TOKEN])[
                                                                                          1:]


class LemmaDataSet(Dataset):
    def __init__(self, data_frame: pd.DataFrame):
        self.data = data_frame

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (str, str):
        row = self.data.iloc[idx]
        return row['word'], row['lemma']

    @property
    def x(self):
        return self.data['word']

    @property
    def y(self):
        return self.data['lemma']


@dataclass
class VocabCreator:
    dataset: LemmaDataSet
    default_token: str = field(default=None)
    special_tokens: List[str] = field(default_factory=set)

    def add_special_token(self, token: str) -> None:
        self.special_tokens.add(token)

    @staticmethod
    def _get_tokens(word_iter):
        yield from list(word_iter)

    def make(self) -> torchtext.vocab.Vocab:
        # check if default token has been set and add it to list of special tokens if it was a case
        specials = self.special_tokens

        vocab = build_vocab_from_iterator(iterator=self._get_tokens(itertools.chain(self.dataset.x, self.dataset.y)),
                                          specials=list(specials),
                                          special_first=False
                                          )
        if not self.default_token:
            vocab.set_default_index(vocab[self.default_token])

        return vocab
