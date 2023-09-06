# Metrics
from abc import ABC
from dataclasses import dataclass
from typing import List, Protocol

import pandas as pd
import torch
from nltk.metrics import distance

Metrics = None

class Metric(ABC):

    @property
    def name(self) -> str:
        return 'Metric'

    def compute(self, target: str, prediction: str) -> float:
        pass



@dataclass
class AccuracyMetric(Metric):

    @property
    def name(self) -> str:
        return 'accuracy'

    def compute(self, target: str, prediction: str) -> float:
        return 1 if target == prediction else 0


@dataclass
class EditDistanceMetric(Metric):

    @property
    def name(self) -> str:
        return 'edit_distance'

    def compute(self, target: str, prediction: str) -> float:
        return distance.edit_distance(target, prediction)


@dataclass
class Metrics:
    _metrics: List[Metric]

    def compute(self, df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
        for metric in self._metrics:
            df[metric.name] = df.apply(lambda row: metric.compute(row[x], row[y]), axis=1)
        return df

    @property
    def metrics(self) -> List[str]:
        return [metric.name for metric in self._metrics]

