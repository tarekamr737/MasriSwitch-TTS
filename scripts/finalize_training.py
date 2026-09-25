"""Create the release train manifest from the selected verified Kaggle stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masriswitch.config import Paths
from masriswitch.train.finalize import finalize_training_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    manifest = finalize_training_manifest(Paths(Path.cwd()), args.result, args.checkpoint)
    print(
        json.dumps(
            {
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "selected_updates": manifest["selected_updates"],
            }
        )
    )


if __name__ == "__main__":
    main()
