import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Tuple, List, Dict
import torch.nn as nn
from transformers import TokenClassificationPipeline

import pandas as pd
from sklearn.metrics import classification_report
from hmmlearn import hmm

EVALUATION_PATH = 'evaluation/evaluations'


class Predictor(Protocol):

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


@dataclass
class HMMPredictor:
    model: hmm.CategoricalHMM

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        data['predictions'] = data.mod_words.apply(lambda l: self.model.predict(l))
        return data
    
@dataclass
class PipelinePredictor:
    pipeline: TokenClassificationPipeline
    
    def _combine_entities_with_average(self, subword_entities):
        grouped_entities = []

        entities = []
        current_entity = ""
        current_tokens = defaultdict(int)
        for entity in subword_entities:
            token = entity['word']
            tag = entity['entity']
            score = entity['score']
            
            if token.startswith('##'):
                current_entity += token[2:]
                current_tokens[tag] += score
            else:
                if current_entity:
                    entities.append((current_entity, max(current_tokens, key=lambda x: current_tokens[x])))
                    
                current_entity = token
                current_tokens.clear()
                current_tokens[tag] = score
                
        if current_entity:
            entities.append((current_entity, max(current_tokens, key=lambda x: current_tokens[x])))
            
        return entities

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        data['predictions'] = data.mod_words.apply(lambda l: [tag for word, tag in self._combine_entities_with_average(self.pipeline(" ".join(l)))])
        return data


@dataclass
class POSEvaluator:
    df: pd.DataFrame
    model_name: str
    path: os.PathLike

    def __evaluate_category(self, data: pd.DataFrame, category: str) -> Dict[str, Dict[str, int]]:

        groupped = data.groupby(category)
        results = {}
        # print(dict(list(groupped)))

        for type, data in dict(list(groupped)).items():
            y_true = [p for l in data.pos_tags.tolist() for p in l]
            y_pred = data.predictions.explode().tolist()
            
            
            if len(y_true) != len(y_pred):
                print(data.mod_words.tolist())
                print(data.pos_tags.tolist())
                print(data.predictions.tolist())
                
            results[type] = classification_report(y_true, y_pred, output_dict=True)
                
        return results

    def evaluate(self, predictor: Predictor, categories: List[str], worst: int = 3) -> None:
        result_df = predictor.predict(self.df)

        eval = {}
        for i, category in enumerate(categories):
            eval[category] = self.__evaluate_category(result_df, category)

        with open(os.path.join(self.path, self.model_name, 'eval.json'), "w") as f:
            json.dump(eval, f, indent=3, ensure_ascii=False)
