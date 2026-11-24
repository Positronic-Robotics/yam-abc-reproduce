import pathlib
import subprocess
import sys

import numpy as np

from openpi import transforms
from openpi.shared import normalize

SCRIPT = pathlib.Path(__file__).with_name("stack_last_command_norm_stats.py")


def test_cli_takes_the_source_and_output_asset_dirs_as_positional_arguments(tmp_path):
    def stats(n):
        return transforms.NormStats(mean=np.arange(n, dtype=float), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))

    normalize.save(tmp_path / "src", {"state": stats(14), "actions": stats(14)})
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "src"), str(tmp_path / "out")], check=True)
    out = normalize.load(tmp_path / "out")
    np.testing.assert_array_equal(out["state"].mean, np.concatenate([np.arange(14)] * 2))
    np.testing.assert_array_equal(out["actions"].mean, np.arange(14))
