from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    train: np.ndarray
    test: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    kind: str = "walk_forward"


def make_fold(number: int, train: np.ndarray, test: np.ndarray, index: pd.DatetimeIndex, *,
              max_train_bars: int | None, embargo: int, kind: str) -> Fold:
    cutoff = int(test.min()) - embargo
    train = train[train < cutoff]
    if max_train_bars:
        train = train[-max_train_bars:]
    if train.size == 0 or test.size == 0:
        raise ValueError("Fold contains an empty train or test segment.")
    return Fold(
        number=number,
        train=train,
        test=test,
        train_start=index[train[0]],
        train_end=index[train[-1]],
        test_start=index[test[0]],
        test_end=index[test[-1]],
        kind=kind)


def outer_folds(index: pd.DatetimeIndex, *, n_folds: int, min_train_bars: int,
                max_train_bars: int | None, embargo: int, holdout_start: str | None = None,
                use_holdout: bool = False) -> tuple[list[Fold], Fold | None]:
    n = len(index)
    holdout_position = (
        int(index.searchsorted(pd.Timestamp(holdout_start))) if use_holdout and holdout_start else n
    )
    development_n = holdout_position
    if development_n <= min_train_bars + n_folds:
        raise ValueError("Insufficient development data for requested folds.")
    boundaries = np.linspace(min_train_bars, development_n, n_folds + 1, dtype=int)
    folds = [
        make_fold(
            number=i,
            train=np.arange(0, boundaries[i]),
            test=np.arange(boundaries[i], boundaries[i + 1]),
            index=index,
            max_train_bars=max_train_bars,
            embargo=embargo,
            kind="walk_forward")
        for i in range(n_folds)
    ]
    holdout = None
    if holdout_position < n:
        holdout = make_fold(
            number=n_folds,
            train=np.arange(0, holdout_position),
            test=np.arange(holdout_position, n),
            index=index,
            max_train_bars=max_train_bars,
            embargo=embargo,
            kind="final_holdout")
    return folds, holdout


def nested_folds(train_positions: np.ndarray, index: pd.DatetimeIndex, *, n_folds: int,
                 min_train_bars: int, max_train_bars: int | None, embargo: int) -> list[Fold]:
    if len(train_positions) <= min_train_bars + n_folds:
        raise ValueError("Insufficient data for inner folds.")
    boundaries = np.linspace(min_train_bars, len(train_positions), n_folds + 1, dtype=int)
    output = []
    for i in range(n_folds):
        train = train_positions[: boundaries[i]]
        test = train_positions[boundaries[i] : boundaries[i + 1]]
        output.append(
            make_fold(
                number=i,
                train=train,
                test=test,
                index=index,
                max_train_bars=max_train_bars,
                embargo=embargo,
                kind="inner")
        )
    return output
