import numpy as np
import pandas as pd
from conftest import market_fixture

from garl_trading.data.dataset import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS, build_features


def test_future_price_change_does_not_modify_past_features():
    raw = market_fixture(("AAA",), seed=7)["AAA"]
    original = build_features(raw)
    changed = raw.copy()
    boundary = 300
    changed.iloc[boundary:, changed.columns.get_indexer(["open", "high", "low", "close"])] *= 3
    modified = build_features(changed)
    pd.testing.assert_frame_equal(
        original.iloc[:boundary].loc[:, FEATURE_COLUMNS],
        modified.iloc[:boundary].loc[:, FEATURE_COLUMNS],
    )


def test_dataset_removes_feature_warmup_before_splitting():
    raw = market_fixture(("AAA", "BBB"), seed=2)
    dataset = build_dataset(raw)
    assert dataset.index[0] == raw["AAA"].index[199]
    assert all(
        np.isfinite(dataset.features[ticker].loc[:, FEATURE_COLUMNS].to_numpy()).all()
        for ticker in dataset.tickers
    )
