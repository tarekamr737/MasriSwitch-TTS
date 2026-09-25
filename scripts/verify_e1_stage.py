"""Check a downloaded E1 stage before spending GPU time on evaluation or resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masriswitch.train.verify import verify_e1_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/kaggle_full_archive.json"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    sha = verify_e1_stage(result, archive, expected_updates=args.updates)
    print(json.dumps({"updates": args.updates, "checkpoint_sha256": sha}))


if __name__ == "__main__":
    main()
