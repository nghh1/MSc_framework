import numpy as np
import torch
from conftest import market_fixture

from garl_trading.data import build_dataset
from garl_trading.data.features import FEATURE_COLUMNS
from garl_trading.rl import (train_independent_a2c, train_independent_dqn, train_independent_ppo,
                             train_joint_a2c, train_joint_dqn, train_joint_ppo)
from garl_trading.rl.core import RewardEarlyStopper, TemporalFeatureExtractor, TradingState
from garl_trading.rl.dqn import Transition, independent_update
from garl_trading.tuning.rl_search import rl_candidate_profiles


def test_rl_temporal_extractor_is_causal_and_preserves_position_state():
    torch.manual_seed(3)
    extractor = TemporalFeatureExtractor(
        observation_size=20 * 3 + 1,
        lookback=20,
        channels=8,
        dropout=0.0).eval()
    original = torch.randn(1, 20, 3)
    changed = original.clone()
    changed[:, 12:, :] += 100

    first = extractor.temporal_states(original)
    second = extractor.temporal_states(changed)
    assert torch.allclose(first[:, :, :12], second[:, :, :12])

    observation = torch.cat([original.reshape(1, -1), torch.tensor([[0.5]])], dim=1)
    encoded = extractor(observation)
    assert encoded.shape == (1, 9)
    assert np.isclose(float(encoded[0, -1].detach()), 0.5)


def test_joint_a2c_ppo_and_dqn_and_independent_ppo_dqn_emit_position_matrices():
    dataset = build_dataset(market_fixture(("AAA", "BBB"), periods=280))
    split = 50
    train_features = {
        ticker: dataset.features[ticker].iloc[:split].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    train_closes = {
        ticker: dataset.prices[ticker]["close"].iloc[:split] for ticker in dataset.tickers
    }
    test_features = {
        ticker: dataset.features[ticker].iloc[split : split + 5].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    context = {
        ticker: dataset.features[ticker].iloc[split - 4 : split].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    common = {
        "levels": (-1.0, 0.0, 1.0),
        "lookback": 5,
        "epochs": 1,
        "rollout_length": 4,
        "learning_rate": 3e-4,
        "gamma": 0.95,
        "cost_rate": 0.0007,
        "seed": 4,
        "decision_interval": 5
    }
    for trainer in (
        train_joint_a2c,
        train_joint_ppo,
        train_independent_ppo,
        train_joint_dqn,
        train_independent_dqn,
    ):
        policy = trainer(train_features, train_closes, **common)
        positions = policy.positions(test_features, context=context)
        assert positions.shape == (5, 2)
        assert positions.abs().max().max() <= 1
        assert (positions.nunique() <= 1).all()
        models = policy.models.values() if isinstance(policy.models, dict) else (policy.models,)
        assert all(hasattr(model, "extractor") for model in models)


def test_early_stopping_checkpointing_starts_after_minimum_epochs():
    model = torch.nn.Linear(1, 1)
    stopper = RewardEarlyStopper(
        patience=2,
        min_delta=0.0,
        minimum_epochs=3)

    assert not stopper.update(0, 1.0, model)
    assert not stopper.update(1, 0.0, model)
    assert not stopper.update(2, 0.0, model)
    assert stopper.best_state is None
    assert stopper.best_epoch is None

    assert not stopper.update(3, 1.0, model)
    assert not stopper.update(4, 0.0, model)
    assert stopper.update(5, 0.0, model)

    assert stopper.best_epoch == 3
    assert stopper.stop_epoch == 5


def test_early_stopping_respects_algorithm_checkpoint_gate():
    model = torch.nn.Linear(1, 1)
    stopper = RewardEarlyStopper(patience=2, min_delta=0.0, minimum_epochs=1)

    assert not stopper.update(0, 1.0, model)
    assert not stopper.update(1, 2.0, model, checkpoint_eligible=False)
    assert stopper.best_state is None

    assert not stopper.update(2, 1.5, model, checkpoint_eligible=True)
    assert stopper.best_epoch == 2


def test_zero_patience_disables_stopping_and_checkpoint_restoration():
    model = torch.nn.Linear(1, 1)
    stopper = RewardEarlyStopper(patience=0, min_delta=0.0, minimum_epochs=1)

    original = model.weight.detach().clone()
    for epoch in range(10):
        assert not stopper.update(epoch, 1.0 - epoch, model)
    with torch.no_grad():
        model.weight.add_(1.0)
    changed = model.weight.detach().clone()
    stopper.restore(model)

    assert stopper.best_state is None
    assert not torch.allclose(original, changed)
    assert torch.allclose(model.weight, changed)


def test_independent_a2c_retains_per_agent_training_diagnostics():
    dataset = build_dataset(market_fixture(("AAA", "BBB"), periods=280))
    features = {
        ticker: dataset.features[ticker].iloc[:50].loc[:, FEATURE_COLUMNS]
        for ticker in dataset.tickers
    }
    closes = {
        ticker: dataset.prices[ticker]["close"].iloc[:50] for ticker in dataset.tickers
    }
    policy = train_independent_a2c(
        features,
        closes,
        levels=(-1.0, 0.0, 1.0),
        lookback=5,
        epochs=1,
        rollout_length=4,
        learning_rate=3e-4,
        gamma=0.95,
        cost_rate=0.0007,
        seed=4)

    assert {row["agent"] for row in policy.diagnostics} == {"AAA", "BBB"}


def test_double_dqn_uses_online_action_and_target_value_with_huber_loss():
    class LookupQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, observation):
            return self.values[observation[:, 0].long()]

    online = LookupQ([[0.0, 0.0], [10.0, 0.0]])
    target = LookupQ([[0.0, 0.0], [1.0, 100.0]])
    optimizer = torch.optim.SGD(online.parameters(), lr=0.01)
    batch = [
        Transition(
            np.array([0.0], dtype=np.float32),
            0,
            0.0,
            np.array([1.0], dtype=np.float32),
            False)
    ]
    loss = independent_update(online, target, optimizer, batch, gamma=1.0)
    assert np.isclose(loss, 0.5)


def test_incremental_hold_action_and_five_day_transition_compound_returns():
    state = TradingState.create(
        np.zeros((7, 1), dtype=np.float32),
        np.asarray([100.0, 110.0, 121.0, 121.0, 121.0, 121.0, 121.0]),
        np.asarray([-1.0, 0.0, 1.0]),
        lookback=1,
        cost_rate=0.0,
        short_borrow_rate=0.0,
        decision_interval=2)
    _, reward, done = state.step(2)
    assert not done
    assert np.isclose(reward, 0.21)
    assert state.cursor == 2
    assert state.target_position == 1.0
    assert state.observation()[-1] == 1.0

    state.step(1)
    assert state.target_position == 1.0
    state.step(0)
    assert state.target_position == 0.0


def test_rl_tuning_profiles_are_bounded_and_include_low_positive_selective_garl_gate():
    for name in (
        "single_a2c",
        "single_ppo",
        "single_dqn",
        "independent_a2c",
        "independent_ppo",
        "independent_dqn",
        "garl_ddal",
        "selective_garl_ddal",
    ):
        profiles = rl_candidate_profiles(name, 3e-4)
        assert len(profiles) == 9
        assert {profile["turnover_penalty_multiplier"] for profile in profiles} == {2.0}
    selective = rl_candidate_profiles("selective_garl_ddal", 3e-4)
    garl = rl_candidate_profiles("garl_ddal", 3e-4)
    assert {profile["pool_size"] for profile in garl + selective} == {3}
    assert {profile["entropy_weight"] for profile in selective} == {0.01}
    assert {
        (profile["learning_rate"], profile["entropy_weight"])
        for profile in garl
    } == {
        (rate, entropy)
        for rate in (1.5e-4, 3e-4, 6e-4)
        for entropy in (0.005, 0.01, 0.02)
    }
    assert {profile["alignment_threshold"] for profile in selective} == {0.0, 0.05, 0.1}
    assert {profile["peer_mix"] for profile in selective} == {0.5}
    assert {
        (profile["learning_rate"], profile["alignment_threshold"])
        for profile in selective
    } == {
        (rate, threshold)
        for rate in (1.5e-4, 3e-4, 6e-4)
        for threshold in (0.0, 0.05, 0.1)
    }
