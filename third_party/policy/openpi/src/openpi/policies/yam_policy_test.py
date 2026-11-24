import dataclasses

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.policies import yam_policy
from openpi.shared import normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

HORIZON = 50
ARM = [*range(6), *range(7, 13)]
GRIPPERS = [6, 13]


@pytest.fixture(autouse=True)
def _no_model_transforms(monkeypatch):
    # The model transforms build the PaliGemma tokenizer, which downloads its model. They are not under test here.
    monkeypatch.setattr(_config.ModelTransformFactory, "__call__", lambda self, model_config: _transforms.Group())


def _data_config(name: str, tmp_path, norm_stats=None) -> _config.DataConfig:
    config = _config.get_config(name)
    if norm_stats is not None:
        normalize.save(tmp_path / "assets" / "stats", norm_stats)
        assets = _config.AssetsConfig(assets_dir=str(tmp_path / "assets"), asset_id="stats")
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, assets=assets))
    else:
        config = dataclasses.replace(config, assets_base_dir=str(tmp_path / "no_assets"))
    return config.data.create(config.assets_dirs, config.model)


def _loader_item(rows: np.ndarray, state: np.ndarray, is_pad: np.ndarray) -> dict:
    """One frame as LeRobot returns it, before the repack transforms."""
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    return {
        "observation.images.top": image,
        "observation.images.left_wrist": image,
        "observation.images.right_wrist": image,
        "observation.state": state,
        "action": rows,
        "action_is_pad": is_pad,
        "prompt": "fold the towel",
    }


def _run(data_config: _config.DataConfig, item: dict) -> dict:
    for transform in [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]:
        item = transform(item)
    return item


def _episode(seed: int = 0, frames: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    actions = np.cumsum(rng.normal(0, 0.02, (frames, 14)), axis=0).astype(np.float32)
    states = (np.roll(actions, 4, axis=0) + rng.normal(0, 0.005, (frames, 14))).astype(np.float32)
    return actions, states


def _window(actions, states, t, history):
    rows = np.stack([actions[min(max(i, 0), len(actions) - 1)] for i in range(t - history, t + HORIZON)])
    is_pad = np.array([i < 0 or i >= len(actions) for i in range(t - history, t + HORIZON)])
    return rows, states[t], is_pad


def test_last_command_state_is_the_action_of_the_frame_before_and_actions_are_relative_to_it(tmp_path):
    data_config = _data_config("pi05_abc130k_last_command", tmp_path)
    assert data_config.action_history == 1
    actions, states = _episode()
    t = 10
    out = _run(data_config, _loader_item(*_window(actions, states, t, history=1)))
    np.testing.assert_allclose(out["state"], actions[t - 1])
    assert out["actions"].shape == (HORIZON, 14)
    np.testing.assert_allclose(
        out["actions"][:, ARM], actions[t : t + HORIZON][:, ARM] - actions[t - 1][ARM], atol=1e-6
    )
    np.testing.assert_allclose(out["actions"][:, GRIPPERS], actions[t : t + HORIZON][:, GRIPPERS])


def test_last_command_at_episode_start_is_the_measured_state(tmp_path):
    data_config = _data_config("pi05_abc130k_last_command", tmp_path)
    actions, states = _episode()
    out = _run(data_config, _loader_item(*_window(actions, states, 0, history=1)))
    np.testing.assert_allclose(out["state"], states[0])
    np.testing.assert_allclose(out["actions"][:, ARM], actions[:HORIZON][:, ARM] - states[0][ARM], atol=1e-6)


def test_last_command_and_measured_state_has_the_last_command_first_and_actions_relative_to_it(tmp_path):
    data_config = _data_config("pi05_abc130k_last_command_and_measured", tmp_path)
    actions, states = _episode()
    t = 10
    out = _run(data_config, _loader_item(*_window(actions, states, t, history=1)))
    np.testing.assert_allclose(out["state"], np.concatenate([actions[t - 1], states[t]]))
    np.testing.assert_allclose(
        out["actions"][:, ARM], actions[t : t + HORIZON][:, ARM] - actions[t - 1][ARM], atol=1e-6
    )


def test_measured_state_config_keeps_the_measured_state_and_no_action_history(tmp_path):
    data_config = _data_config("pi05_abc130k", tmp_path)
    assert data_config.action_history == 0
    actions, states = _episode()
    t = 10
    rows, state, _ = _window(actions, states, t, history=0)
    item = _loader_item(rows, state, np.zeros(HORIZON, dtype=bool))
    del item["action_is_pad"]
    out = _run(data_config, item)
    np.testing.assert_allclose(out["state"], states[t])
    np.testing.assert_allclose(out["actions"][:, ARM], actions[t : t + HORIZON][:, ARM] - states[t][ARM], atol=1e-6)


def _request(state: np.ndarray, last_command: np.ndarray | None) -> dict:
    request = {
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": "fold the towel",
    }
    if last_command is not None:
        request[yam_policy.LAST_COMMAND_KEY] = last_command
    return request


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pi05_abc130k_last_command", lambda last, measured: last),
        ("pi05_abc130k_last_command_and_measured", lambda last, measured: np.concatenate([last, measured])),
    ],
)
def test_served_request_takes_the_state_from_the_last_command_it_sends(tmp_path, name, expected):
    data_config = _data_config(name, tmp_path)
    measured, last = np.full(14, 0.1, dtype=np.float32), np.full(14, 0.2, dtype=np.float32)
    out = _run(dataclasses.replace(data_config, repack_transforms=_transforms.Group()), _request(measured, last))
    np.testing.assert_allclose(out["state"], expected(last, measured))
    with pytest.raises(KeyError, match="last_command"):
        _run(dataclasses.replace(data_config, repack_transforms=_transforms.Group()), _request(measured, None))


class _FakeMeta:
    fps = 30

    def __init__(self, repo_id, root=None):
        self.tasks = {0: "fold the towel"}


class _FakeLeRobotDataset:
    delta_timestamps = None

    def __init__(self, repo_id, root=None, tolerance_s=None, delta_timestamps=None):
        type(self).delta_timestamps = delta_timestamps


@pytest.mark.parametrize(("name", "first_offset"), [("pi05_abc130k_last_command", -1), ("pi05_abc130k", 0)])
def test_loader_reads_one_action_before_the_frame_only_for_the_last_command_state(
    tmp_path, monkeypatch, name, first_offset
):
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMeta)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    config = _config.get_config(name)
    _data_loader.create_torch_dataset(_data_config(name, tmp_path), config.model.action_horizon, config.model)
    offsets = [round(t * 30) for t in _FakeLeRobotDataset.delta_timestamps["action"]]
    assert offsets == list(range(first_offset, HORIZON))


def _norm_stats(dim: int) -> dict[str, _transforms.NormStats]:
    def stats(n):
        return _transforms.NormStats(
            mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.arange(1, n + 1, dtype=np.float64)
        )

    return {"state": stats(dim), "actions": stats(14)}


def test_last_command_and_measured_state_needs_28_dim_norm_stats(tmp_path):
    with pytest.raises(ValueError, match="28"):
        _data_config("pi05_abc130k_last_command_and_measured", tmp_path, norm_stats=_norm_stats(14))
    stacked = yam_policy.stack_last_command_norm_stats(_norm_stats(14))
    np.testing.assert_allclose(stacked["state"].q99, np.concatenate([np.arange(1, 15)] * 2))
    np.testing.assert_allclose(stacked["actions"].q99, np.arange(1, 15))
    data_config = _data_config("pi05_abc130k_last_command_and_measured", tmp_path / "b", norm_stats=stacked)
    assert data_config.norm_stats["state"].q99.shape == (28,)
    _data_config("pi05_abc130k_last_command", tmp_path / "a", norm_stats=_norm_stats(14))
