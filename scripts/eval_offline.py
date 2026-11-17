"""Offline evaluation of ForceLDM checkpoints on held-out episodes.

For frames sampled from the validation split, the student policy receives only what is available at deployment
(current RGB views, state, wrench history, prompt) and predicts a 32-step action chunk. The prediction is compared
with the recorded action chunk. A "hold current state" baseline is reported for reference.

Example:
    python scripts/eval_offline.py --config pi0_forceldm_lora \
        --checkpoints checkpoints/pi0_forceldm_lora/run/1000 checkpoints/pi0_forceldm_lora/run/4999 \
        --frame-stride 30 --output results/eval_val.json
"""

import argparse
import collections
import dataclasses
import json
import logging
import pathlib
import time

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

STATE_DIM = 7
HORIZON_BUCKETS = {"t1-8": slice(0, 8), "t9-16": slice(8, 16), "t17-32": slice(16, 32)}


def _to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _image_hwc_uint8(img) -> np.ndarray:
    img = _to_numpy(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img


def _summarize(errors: list[np.ndarray]) -> dict:
    """errors: list of (32, 7) arrays of pred - gt."""
    e = np.stack(errors)  # (N, H, D)
    out = {
        "n": int(e.shape[0]),
        "mse": float(np.mean(e**2)),
        "l1": float(np.mean(np.abs(e))),
        "l1_per_dim": np.mean(np.abs(e), axis=(0, 1)).round(5).tolist(),
        "l1_first_step": float(np.mean(np.abs(e[:, 0]))),
    }
    for name, sl in HORIZON_BUCKETS.items():
        out[f"l1_{name}"] = float(np.mean(np.abs(e[:, sl])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_forceldm_lora")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--frame-stride", type=int, default=30, help="Evaluate every N-th frame of each episode.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-episodes", type=int, nargs="*", default=None, help="Only evaluate these episodes.")
    parser.add_argument("--exclude-episodes", type=int, nargs="*", default=None, help="Skip these episodes.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    train_config = _config.get_config(args.config)
    data_factory = dataclasses.replace(train_config.data, split=args.split)
    train_config = dataclasses.replace(train_config, data=data_factory)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    n_hist = int(train_config.model.force_input_frames)

    indices = list(range(0, len(dataset), args.frame_stride))
    if args.include_episodes is not None or args.exclude_episodes:
        # Filter by episode via the numeric column (no video decoding); stride positions stay identical.
        base = dataset
        while not hasattr(base, "hf_dataset"):
            base = base._dataset
        ep_col = base.hf_dataset.with_format("numpy")["episode_index"]
        keep = set(args.include_episodes) if args.include_episodes is not None else None
        drop = set(args.exclude_episodes or [])
        indices = [i for i in indices if (keep is None or int(ep_col[i]) in keep) and int(ep_col[i]) not in drop]
    if args.max_samples is not None and len(indices) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(indices, size=args.max_samples, replace=False).tolist())
    logging.info("Split %s: %d frames, evaluating %d samples", args.split, len(dataset), len(indices))

    # Load the samples once so every checkpoint sees identical inputs.
    samples = []
    for i in indices:
        item = dataset[i]
        effort = _to_numpy(item["observation.effort"]).astype(np.float32)
        obs = {
            cam: _image_hwc_uint8(item[cam]) for cam in data_factory.camera_mapping.values()
        }
        obs["observation.state"] = _to_numpy(item["observation.state"]).astype(np.float32)
        obs["observation.effort"] = effort[:n_hist]  # history/current only, as at deployment
        obs["prompt"] = item["prompt"]
        gt = _to_numpy(item["action"]).astype(np.float32)[:, :STATE_DIM]
        samples.append((obs, gt, item["prompt"], int(item["episode_index"])))
    logging.info("Loaded %d samples", len(samples))

    results = {
        "config": args.config,
        "split": args.split,
        "frame_stride": args.frame_stride,
        "include_episodes": args.include_episodes,
        "exclude_episodes": args.exclude_episodes,
        "models": {},
    }
    per_sample = {"episode": np.array([s[3] for s in samples]), "prompt": np.array([s[2] for s in samples])}

    # Baseline: hold the current state for the whole chunk.
    hold = collections.defaultdict(list)
    for obs, gt, prompt, _ in samples:
        err = obs["observation.state"][:STATE_DIM][None, :] - gt
        hold["all"].append(err)
        hold[prompt].append(err)
    results["models"]["baseline_hold_state"] = {k: _summarize(v) for k, v in hold.items()}
    per_sample["baseline_hold_state"] = np.stack(hold["all"])

    for ckpt in args.checkpoints:
        logging.info("Evaluating %s", ckpt)
        policy = _policy_config.create_trained_policy(train_config, ckpt)
        errs = collections.defaultdict(list)
        noise_rng = np.random.default_rng(args.seed)
        t0 = time.time()
        for obs, gt, prompt, _ in samples:
            noise = noise_rng.standard_normal(
                (train_config.model.action_horizon, train_config.model.action_dim)
            ).astype(np.float32)
            pred = policy.infer(dict(obs), noise=noise)["actions"][:, :STATE_DIM]
            err = pred - gt
            errs["all"].append(err)
            errs[prompt].append(err)
        results["models"][str(ckpt)] = {k: _summarize(v) for k, v in errs.items()}
        per_sample[str(ckpt)] = np.stack(errs["all"])
        results["models"][str(ckpt)]["all"]["sec_per_sample"] = (time.time() - t0) / len(samples)
        logging.info("%s: %s", ckpt, results["models"][str(ckpt)]["all"])
        del policy

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    # Signed per-sample errors (pred - gt), shape (N, horizon, 7) per model, for later re-aggregation.
    np.savez_compressed(output.with_suffix(".errors.npz"), **per_sample)

    print(f"\n{'model':60s} {'n':>5s} {'MSE':>10s} {'L1':>8s} {'L1@t1':>8s} {'L1 t1-8':>8s} {'L1 t17-32':>9s}")
    for name, per in results["models"].items():
        a = per["all"]
        print(f"{name[-60:]:60s} {a['n']:5d} {a['mse']:10.5f} {a['l1']:8.4f} {a['l1_first_step']:8.4f} "
              f"{a['l1_t1-8']:8.4f} {a['l1_t17-32']:9.4f}")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
