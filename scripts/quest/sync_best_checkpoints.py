#!/usr/bin/env python3
"""Copy the best validation checkpoint from completed Quest runs to local storage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REMOTE_SCAN = r'''
import json
import os
import re
import sys

root = os.path.realpath(sys.argv[1])
required_epochs = int(sys.argv[2])
epoch_re = re.compile(r"^epoch=(\d+)(?:-|\.ckpt$)")

for current, dirs, files in os.walk(root):
    if "run-metadata.txt" not in files:
        continue
    meta_path = os.path.join(current, "run-metadata.txt")
    metadata = {}
    with open(meta_path, encoding="utf-8") as handle:
        for line in handle:
            key, sep, value = line.strip().partition("=")
            if sep:
                metadata[key] = value
    if metadata.get("epochs") != str(required_epochs):
        continue
    if not os.path.isfile(os.path.join(current, "SUCCESS")):
        continue

    rel = os.path.relpath(current, root).split(os.sep)
    if len(rel) < 3:
        continue
    dataset, model, run_id = rel[-3:]
    checkpoint_dir = os.path.join(current, "checkpoints")
    log_path = os.path.join(current, "logs.json.txt")
    if not os.path.isdir(checkpoint_dir) or not os.path.isfile(log_path):
        continue

    checkpoints = {}
    for name in os.listdir(checkpoint_dir):
        match = epoch_re.match(name)
        if match and name.endswith(".ckpt"):
            checkpoints.setdefault(int(match.group(1)), []).append(name)

    validation = {}
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if "epoch" in row and "val_loss" in row:
                    validation[int(row["epoch"])] = float(row["val_loss"])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

    candidates = [(loss, epoch) for epoch, loss in validation.items()
                  if epoch in checkpoints]
    if not candidates:
        continue
    loss, epoch = min(candidates)
    names = checkpoints[epoch]
    exact = "epoch={:04d}.ckpt".format(epoch)
    names.sort(key=lambda name: (
        0 if "val_loss=" in name else 1 if name == exact else 2,
        name,
    ))
    checkpoint = os.path.join(checkpoint_dir, names[0])
    print(json.dumps({
        "dataset": dataset,
        "policy": metadata.get("policy", "unknown"),
        "model": model,
        "variant": metadata.get("variant", ""),
        "run_id": run_id,
        "epochs": required_epochs,
        "best_epoch": epoch,
        "val_loss": loss,
        "checkpoint": checkpoint,
        "size": os.path.getsize(checkpoint),
        "mtime": os.path.getmtime(meta_path),
    }, sort_keys=True))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find successful Quest runs, select the lowest validation-loss "
            "checkpoint, and copy it into a dataset/policy/model/run hierarchy."
        )
    )
    parser.add_argument(
        "--host", default="xph8283@login.quest.northwestern.edu",
        help="Quest SSH host (default: %(default)s)",
    )
    parser.add_argument(
        "--remote-root",
        default="/projects/p52914/xph8283/polyumi/outputs",
        help="Quest output root (default: %(default)s)",
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument(
        "--model", action="append", default=[],
        help="Only export this model name; repeat for multiple models",
    )
    parser.add_argument(
        "--expect", default=0, type=int,
        help="Fail unless this many dataset/model checkpoints are found",
    )
    parser.add_argument(
        "--all-runs", action="store_true",
        help="Export every matching completed run instead of only the newest per dataset/model",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scan_remote(args: argparse.Namespace) -> list[dict]:
    remote_command = " ".join([
        "python3", "-c", shlex.quote(REMOTE_SCAN),
        shlex.quote(args.remote_root), shlex.quote(str(args.epochs)),
    ])
    command = [
        "ssh", "-o", "BatchMode=yes", args.host, remote_command,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise SystemExit(f"Quest scan failed with exit code {result.returncode}")
    runs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if args.model:
        wanted = set(args.model)
        runs = [run for run in runs if run["model"] in wanted]
    if not args.all_runs:
        newest = {}
        for run in runs:
            key = (run["dataset"], run["policy"], run["model"])
            if key not in newest or run["mtime"] > newest[key]["mtime"]:
                newest[key] = run
        runs = list(newest.values())
    runs.sort(key=lambda run: (
        run["dataset"], run["policy"], run["model"], run["run_id"]
    ))
    if args.expect and len(runs) != args.expect:
        raise SystemExit(
            f"Expected {args.expect} checkpoints, found {len(runs)}. "
            "Nothing was copied."
        )
    return runs


def copy_checkpoint(args: argparse.Namespace, run: dict) -> Path:
    output_dir = (
        args.destination / run["dataset"] / run["policy"] / run["model"]
        / f"run-{run['run_id']}"
    )
    loss_text = format(run["val_loss"], ".10g")
    target = output_dir / (
        f"best-epoch={run['best_epoch']:04d}-val_loss={loss_text}.ckpt"
    )
    if args.dry_run:
        return target

    output_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == run["size"]:
        print(f"Already present: {target}")
    else:
        partial = target.with_name(target.name + ".partial")
        source = f"{args.host}:{run['checkpoint']}"
        if shutil.which("rsync"):
            command = [
                "rsync", "--partial", "--info=progress2", "-e",
                "ssh -o BatchMode=yes", source, str(partial),
            ]
        else:
            command = ["scp", "-o", "BatchMode=yes", source, str(partial)]
        subprocess.run(command, check=True)
        if partial.stat().st_size != run["size"]:
            raise RuntimeError(f"Size verification failed for {partial}")
        os.replace(partial, target)

    selection = dict(run)
    selection["local_checkpoint"] = str(target)
    metadata_path = output_dir / "selection.json"
    metadata_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return target


def write_manifest(destination: Path, rows: list[tuple[dict, Path]]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "best-checkpoints.tsv"
    lines = [
        "dataset\tpolicy\tmodel\trun_id\tepoch\tval_loss\tcheckpoint"
    ]
    for run, target in rows:
        lines.append(
            "\t".join([
                run["dataset"], run["policy"], run["model"], run["run_id"],
                str(run["best_epoch"]), repr(run["val_loss"]), str(target),
            ])
        )
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    print(f"Manifest: {manifest}")


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.expect < 0:
        raise SystemExit("--epochs must be positive and --expect cannot be negative")
    runs = scan_remote(args)
    if not runs:
        raise SystemExit("No matching successful runs with validation checkpoints were found")

    print(f"Selected {len(runs)} checkpoint(s):")
    copied = []
    for run in runs:
        target = copy_checkpoint(args, run)
        copied.append((run, target))
        print(
            f"  {run['dataset']}/{run['policy']}/{run['model']} "
            f"run={run['run_id']} epoch={run['best_epoch']} "
            f"val_loss={run['val_loss']:.10g} -> {target}"
        )
    if not args.dry_run:
        write_manifest(args.destination, copied)
    else:
        print("Dry run: nothing copied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
