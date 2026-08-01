import json

from conftest import market_fixture

from garl_trading.config import FrameworkConfig, ModelsConfig
from garl_trading.experiment.artifacts import ArtifactStore


def test_artifact_manifest_hashes_config_and_market_snapshot(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'test'\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "artifacts", "test")
    store.initialise(FrameworkConfig(), config_path)
    store.save_market_data(market_fixture(("AAA",), periods=10))
    manifest = json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["data_sha256"]) == 64


def test_cuda_index_is_a_valid_configured_device():
    config = FrameworkConfig(models=ModelsConfig(device="cuda:1"))
    config.validate()
