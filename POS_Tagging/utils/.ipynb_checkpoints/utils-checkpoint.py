import numpy as np
import pandas as pd
from typing import Dict


def get_initial_prob(df: pd.DataFrame, state_col: str, laplace_smoothing: int = 0) -> Dict[str, float]:
    # extract (index, state) pairs
    pairs = [(state, idx) for states in df[state_col] for idx, state in enumerate(states)]

    # count every (state, index) pair and add laplace_smoothing
    statistics = pd.DataFrame(pairs, columns=[state_col, 'index'])
    probabilities = statistics.pivot_table(columns='index', index=state_col, aggfunc=np.size).fillna(0).applymap(
        lambda cnt: cnt + laplace_smoothing)

    # normalize
    probabilities /= probabilities.apply(lambda col: col.sum())

    return probabilities.to_dict()[0]


def get_transitions(df: pd.DataFrame, state_col: str, laplace_smoothing: int = 0) -> Dict[str, Dict[str, float]]:

    # extract every (state[i], state[i+1]) pair
    pairs = [(states[idx], states[+ 1]) for states in df[state_col] for idx in range(len(states) - 1)]

    # count every (state[i], state[i+1]) pair and add laplace_smoothing
    statistics = pd.DataFrame(pairs, columns=['from', 'to'])
    probabilities = statistics.pivot_table(columns='from', index='to', aggfunc=np.size).fillna(0).applymap(
        lambda cnt: cnt + laplace_smoothing)

    # normalize
    probabilities /= probabilities.apply(lambda col: col.sum())

    return probabilities.to_dict()


def get_emissions(df: pd.DataFrame, state_col: str, emission_col: str, laplace_smoothing: int = 0) -> Dict[
    str, Dict[str, float]]:
    # extract each pair of (state -> emission)
    pairs = [(state, emission) for states, emissions in zip(df[state_col].iloc, df[emission_col].iloc) for
             state, emission in zip(states, emissions)]

    # count every (state, emission) pair and add laplace_smoothing
    emissions = pd.DataFrame(pairs, columns=[state_col, emission_col]).pivot_table(columns=state_col,
                                                                                   index=emission_col,
                                                                                   aggfunc=np.size).fillna(0).applymap(
        lambda cnt: cnt + laplace_smoothing)

    # normalize
    emissions /= emissions.apply(lambda col: col.sum() )

    return emissions.to_dict()
