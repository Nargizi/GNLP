import os.path
import json
from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import List, Dict

import pandas as pd
import torchtext
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader

from utils.constants import PAD_TOKEN, EOW_TOKEN, UNK_TOKEN, SOW_TOKEN
from utils.dataset import coallate_words, tensors_to_words, LemmaDataSet, VocabCreator
from evaluation.metrics import Metrics

EVALUATION_PATH = 'evaluation/evaluations'


@dataclass
class LemmatizatorEvaluator:
    df: pd.DataFrame
    metrics: Metrics
    path: PathLike
    model_name: str
    max_length: int

    @cached_property
    def vocab(self) -> torchtext.vocab.Vocab:

        return VocabCreator(LemmaDataSet(self.df), default_token=PAD_TOKEN,
                            special_tokens=[EOW_TOKEN, UNK_TOKEN, PAD_TOKEN, SOW_TOKEN]).make()

    def __evaluate_category(self, data, category, fig, axs, worst=3) -> Dict[str, Dict[str, int]]:

        groupped = data.groupby(category)
        df = pd.DataFrame()
        for metric in self.metrics.metrics:
            df[metric] = groupped[metric].mean()

        df.reset_index(inplace=True)
        for i, metric in enumerate(self.metrics.metrics):
            axs[i + 1].set_title(f'Avg. {metric}')
            sns.barplot(df.sort_values(by=metric, ascending=True), x=category, y=metric, ax=axs[i + 1])

        df_melted = df.melt(id_vars=category, var_name='metric', value_name='value', value_vars=self.metrics.metrics)
        sns.barplot(df_melted, x=category, y='value', hue='metric', ax=axs[0])
        axs[0].set_title(f'{category}')

        values = df.set_index(category).to_dict('index')

        for c in data[category].unique():
            worst_preds = data[data[category] == c].sort_values(by='edit_distance', ascending=False)[
                ['word', 'lemma', 'pred', 'edit_distance']]

            candidates = {}
            for i, row in enumerate(worst_preds.values.tolist()[:worst]):
                candidates[i + 1] = {'distance': row[3], 'word': row[0], 'lemma': row[1], 'prediction': row[2]}

            values[c][f'top_{worst}_worst'] = candidates

        return values

    def __get_predictions(self, model: nn.Module, loader) -> pd.DataFrame:
        pred = {}
        for i, (x, y) in enumerate(loader):
            y_pred = model(y).argmax(dim=2)
            x = tensors_to_words(x, self.vocab)
            y_pred = tensors_to_words(y_pred, self.vocab)
            for x_i, y_pred_i in zip(x, y_pred):
                pred[x_i] = y_pred_i
        return pd.DataFrame.from_dict(pred, orient='index', columns=['pred'])

    def evaluate(self, model: nn.Module, categories: List[str], worst: int = 3) -> None:
        loader = DataLoader(LemmaDataSet(self.df), batch_size=512, collate_fn=lambda batch: coallate_words(batch,
                                                                                                           self.max_length,
                                                                                                           self.vocab))
        pred_df = self.__get_predictions(model, loader)
        result_df = self.df.set_index('word').merge(pred_df, left_index=True, right_index=True)
        result_df = self.metrics.compute(result_df, 'lemma', 'pred').reset_index()
        result_df = result_df.rename({'index': 'word'}, axis=1)

        fig, axs = plt.subplots(nrows=len(self.metrics.metrics) + 1, ncols=len(categories), figsize=(60, 75))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)

        eval = {}
        for i, category in enumerate(categories):
            eval[category] = self.__evaluate_category(result_df, category, fig, axs[:, i], worst)

        plt.savefig(os.path.join(self.path, self.model_name, 'plots.png'))

        with open(os.path.join(self.path, self.model_name, 'values.json'), "w") as f:
            json.dump(eval, f, indent=3, ensure_ascii=False)
