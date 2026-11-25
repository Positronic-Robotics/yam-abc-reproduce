import numpy as np
import pytest

from yam_abc_reproduce.deploy.servers.openpi_server import LAST_COMMAND, RemapPolicy


class FakePolicy:
    def __init__(self):
        self.seen = None

    def infer(self, obs):
        self.seen = obs
        return {"actions": np.zeros((1, 14))}


def _request(**extra):
    return {
        "images": {"top": np.zeros((2, 2, 3), np.uint8)},
        "state": np.ones(14),
        "prompt": "p",
        **extra,
    }


def test_the_last_command_goes_to_its_model_key():
    policy = FakePolicy()
    RemapPolicy(policy, {}, "observation/", "observation/state", "observation/last_command").infer(
        _request(**{LAST_COMMAND: np.full(14, 2.0)})
    )
    np.testing.assert_array_equal(
        policy.seen["observation/last_command"], np.full(14, 2.0, np.float32)
    )
    np.testing.assert_array_equal(policy.seen["observation/state"], np.ones(14, np.float32))


def test_a_last_command_config_refuses_a_request_without_one():
    with pytest.raises(KeyError, match=LAST_COMMAND):
        RemapPolicy(
            FakePolicy(), {}, "observation/", "observation/state", "observation/last_command"
        ).infer(_request())


def test_without_a_last_command_key_the_request_field_does_not_reach_the_model():
    policy = FakePolicy()
    RemapPolicy(policy, {}, "observation/", "observation/state").infer(
        _request(**{LAST_COMMAND: np.full(14, 2.0)})
    )
    assert set(policy.seen) == {"observation/state", "prompt", "observation/top"}
