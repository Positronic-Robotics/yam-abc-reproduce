"""Policy transforms for YAM LeRobot datasets.

YAM episodes are converted to LeRobot v3.0 with:
  observation.state              (14,) = [L: 6 joints + 1 gripper, R: 6 joints + 1 gripper]
  action                         (14,) same layout (leader-derived command)
  observation.images.top_rgb     third-person (overhead)
  observation.images.left_rgb    left view   -> left wrist slot
  observation.images.right_rgb   right view  -> right wrist slot

The model state is selected by `StateInput`. The last command at frame t is the action of frame t-1: the
target that was sent to the arms before the observation of frame t.
"""

import dataclasses
import enum

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

YAM_ACTION_DIM = 14  # 2 arms x (6 joints + 1 gripper)

# Request key of the last command sent to the arms, in the layout of observation/state.
LAST_COMMAND_KEY = "observation/last_command"


class StateInput(enum.Enum):
    """What the model gets as its state input."""

    # The measured follower joints (14-D).
    MEASURED = "measured"
    # The last command sent to the arms (14-D).
    LAST_COMMAND = "last_command"
    # The last command, then the measured joints (28-D). The last command is first, so that DeltaActions and
    # AbsoluteActions, which use the first 14 state dims, take the actions relative to the last command.
    LAST_COMMAND_AND_MEASURED = "last_command_and_measured"

    @property
    def uses_last_command(self) -> bool:
        return self is not StateInput.MEASURED

    @property
    def dim(self) -> int:
        return 2 * YAM_ACTION_DIM if self is StateInput.LAST_COMMAND_AND_MEASURED else YAM_ACTION_DIM


def make_yam_example() -> dict:
    """Random input example (matches the repacked key names)."""
    return {
        "observation/state": np.random.rand(YAM_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores video as (C, H, W)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LastCommandFromActionHistory(transforms.DataTransformFn):
    """Training only. Split the action row of frame t-1 off the action sequence as the last command.

    Needs a data loader with action_history=1: `actions` has action_horizon + 1 rows and `actions_is_pad`
    marks the rows outside the episode. At the first frame of an episode no command was sent yet, so the last
    command is the measured state. A serving client sends the measured state there for the same reason.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        is_pad = np.asarray(data.pop("actions_is_pad"))
        source = data["observation/state"] if is_pad[0] else actions[0]
        data[LAST_COMMAND_KEY] = np.array(source, dtype=np.float32)
        data["actions"] = actions[1:]
        return data


@dataclasses.dataclass(frozen=True)
class YamStateInput(transforms.DataTransformFn):
    """Set observation/state to the model state that `state_input` selects (train + inference)."""

    state_input: StateInput

    def __call__(self, data: dict) -> dict:
        if not self.state_input.uses_last_command:
            return data
        if LAST_COMMAND_KEY not in data:
            raise KeyError(f"{LAST_COMMAND_KEY} is missing: the state input {self.state_input.value} needs it")
        last_command = np.asarray(data.pop(LAST_COMMAND_KEY), dtype=np.float32)
        if self.state_input is StateInput.LAST_COMMAND:
            data["observation/state"] = last_command
        else:
            measured = np.asarray(data["observation/state"], dtype=np.float32)
            data["observation/state"] = np.concatenate([last_command, measured], axis=-1)
        return data


def stack_last_command_norm_stats(norm_stats: dict[str, transforms.NormStats]) -> dict[str, transforms.NormStats]:
    """Norm stats for StateInput.LAST_COMMAND_AND_MEASURED: the 14-D state stats for each half of the state.

    The commanded joints and grippers span the same range as the measured ones, so the state stats serve both.
    """
    state = norm_stats["state"]
    if state.mean.shape[-1] != YAM_ACTION_DIM:
        raise ValueError(f"state norm stats have {state.mean.shape[-1]} dims, expected {YAM_ACTION_DIM}")

    def twice(x):
        return None if x is None else np.concatenate([x, x], axis=-1)

    stacked = transforms.NormStats(
        mean=twice(state.mean), std=twice(state.std), q01=twice(state.q01), q99=twice(state.q99)
    )
    return {**norm_stats, "state": stacked}


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Convert repacked YAM inputs into the model's expected format (train + inference)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        left = _parse_image(data["observation/left_wrist"])
        right = _parse_image(data["observation/right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:  # only present during training
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Strip model action padding back to YAM's real dims (inference only).

    ``action_dim`` is 7 per arm (6 joints + 1 gripper): 14 bimanual, 7 single-arm."""

    action_dim: int = YAM_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
