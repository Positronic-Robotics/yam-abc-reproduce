"""Write the norm stats of a YAM asset for the state input last_command_and_measured (28-D state).

The state stats of the source asset (14-D) serve for both halves of the state. The action stats do not change.

Usage:
  python scripts/stack_last_command_norm_stats.py SRC_ASSET_DIR OUT_ASSET_DIR
  e.g. assets/yam assets/yam_last_command_and_measured
"""

import argparse
import logging
import pathlib

import openpi.policies.yam_policy as yam_policy
import openpi.shared.normalize as normalize

logger = logging.getLogger(__name__)


def main(src: pathlib.Path, out: pathlib.Path) -> None:
    stacked = yam_policy.stack_last_command_norm_stats(normalize.load(src))
    normalize.save(out, stacked)
    logger.info("wrote %s: state %d dims", out, stacked["state"].mean.shape[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=pathlib.Path, help="asset dir with the 14-D norm stats")
    parser.add_argument("out", type=pathlib.Path, help="asset dir for the 28-D norm stats")
    args = parser.parse_args()
    main(args.src, args.out)
