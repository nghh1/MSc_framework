import numpy as np
import pandas as pd

from garl_trading.validation import outer_folds


def test_walk_forward_is_purged_capped_and_has_untouched_holdout():
    index = pd.bdate_range("2010-01-01", periods=3000)
    folds, holdout = outer_folds(
        index,
        n_folds=4,
        min_train_bars=500,
        max_train_bars=700,
        embargo=2,
        holdout_start=str(index[2500].date()),
        use_holdout=True,
    )
    assert len(folds) == 4
    assert holdout is not None
    assert holdout.test[0] == 2500
    for fold in [*folds, holdout]:
        assert len(fold.train) <= 700
        assert fold.train[-1] <= fold.test[0] - 3
        assert not np.intersect1d(fold.train, fold.test).size
