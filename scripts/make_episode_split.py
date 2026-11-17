"""Create a per-task train/val episode split for a LeRobot v2.1 dataset.

Example:
    python scripts/make_episode_split.py --repo-id PeiYuan001/forceldm-dataset \
        --val-fraction 0.05 --seed 0 --output splits/forceldm-dataset_val5_seed0.json
"""

import argparse
import collections
import json
import math
import pathlib
import random

from lerobot.common.constants import HF_LEROBOT_HOME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    episodes_file = pathlib.Path(HF_LEROBOT_HOME) / args.repo_id / "meta" / "episodes.jsonl"
    by_task: dict[str, list[int]] = collections.defaultdict(list)
    lengths: dict[int, int] = {}
    with episodes_file.open() as f:
        for line in f:
            ep = json.loads(line)
            by_task[ep["tasks"][0]].append(int(ep["episode_index"]))
            lengths[int(ep["episode_index"])] = int(ep["length"])

    rng = random.Random(args.seed)
    train, val, per_task = [], [], {}
    for task in sorted(by_task):
        eps = sorted(by_task[task])
        n_val = max(1, math.ceil(len(eps) * args.val_fraction))
        held = sorted(rng.sample(eps, n_val))
        val += held
        train += [e for e in eps if e not in held]
        per_task[task] = {"total": len(eps), "val": held}

    out = {
        "repo_id": args.repo_id,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train": sorted(train),
        "val": sorted(val),
        "per_task": per_task,
        "frames": {"train": sum(lengths[e] for e in train), "val": sum(lengths[e] for e in val)},
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"train={len(train)} episodes / {out['frames']['train']} frames; "
          f"val={len(val)} episodes / {out['frames']['val']} frames -> {output}")
    for task, info in per_task.items():
        print(f"  {info['total']:4d} eps, val {info['val']}: {task}")


if __name__ == "__main__":
    main()
