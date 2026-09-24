#!/usr/bin/env python3
"""
Find which test frames the texture classifier gets wrong, and whether they cluster.

Retrains the same refit model the report scores (same seed, same budget), predicts the held-out
test frames, and breaks the errors down three ways for the classes asked about:

* per SESSION -- a class that fails as whole recordings is a different problem from one that
  fails as scattered frames. The first says some grasps are unlike the training ones; the second
  says particular moments within a grasp are.
* per POSITION within the recording -- frames are sampled evenly across a session, so position
  stands in for time: early frames are the approach, late ones the release, and neither has the
  object pressed the way the middle does.
* per BRIGHTNESS -- a proxy for whether anything is in contact at all.

It also writes contact sheets of the worst frames beside correctly classified ones from the same
class, because the numbers cannot say what is actually in the picture.

Usage:
    /usr/bin/python3 analysis/texture_error_analysis.py --tensors texture_ranges.npz \
        --classes triangle circle --out-dir /tmp/texture_errors
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_texture_kfold import load, predict, train_model  # noqa: E402


def positions_within_session(sessions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Fractional position of each frame inside its own session, 0.0 first to 1.0 last.

    Frames were written per session in temporal order, so a frame's rank among its session's rows
    is its rank in time. Expressed as a fraction because sessions differ in length.
    """
    out = np.zeros(len(indices), dtype=float)
    for session in np.unique(sessions[indices]):
        rows = np.flatnonzero(sessions[indices] == session)
        out[rows] = np.linspace(0.0, 1.0, len(rows)) if len(rows) > 1 else 0.5
    return out


def tile(images: np.ndarray, per_row: int = 8) -> np.ndarray:
    """Lay frames out in a grid, padding the last row so the grid stays rectangular."""
    if len(images) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    rows = []
    for start in range(0, len(images), per_row):
        chunk = list(images[start : start + per_row])
        while len(chunk) < per_row:
            chunk.append(np.zeros_like(images[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    """Train, predict, and break the errors down."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tensors', type=pathlib.Path, default=pathlib.Path('texture_ranges.npz'))
    ap.add_argument('--classes', nargs='+', default=['triangle', 'circle'])
    ap.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('/tmp/texture_errors'))
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    args.augment = True
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y, split, sessions, classes = load(args.tensors, args.device)
    raw = np.load(args.tensors, allow_pickle=False)['images']
    y_np = y.cpu().numpy()
    pool = np.flatnonzero(split == 'pool')
    test = np.flatnonzero(split == 'test')

    model, _, _ = train_model(
        x,
        y,
        torch.as_tensor(pool, device=args.device),
        None,
        SimpleNamespace(**vars(args)),
        args.device,
        len(classes),
        seed=args.seed,
    )
    pred = predict(model, x, torch.as_tensor(test, device=args.device), args.batch)
    correct = pred == y_np[test]
    print(f'\noverall test accuracy {correct.mean():.3f} on {len(test)} frames\n')

    position = positions_within_session(sessions, test)
    brightness = raw[test].reshape(len(test), -1).mean(axis=1)

    for name in args.classes:
        ci = classes.index(name)
        rows = np.flatnonzero(y_np[test] == ci)
        ok = correct[rows]
        print(f'===== {name}: {ok.sum()}/{len(rows)} correct ({ok.mean():.3f}) ' + '=' * 20)

        # --- per session ---
        print('  per session (8 held-out recordings, 20 frames each):')
        for session in sorted(set(sessions[test][rows])):
            in_sess = rows[sessions[test][rows] == session]
            acc = correct[in_sess].mean()
            wrong_as = [classes[p] for p in pred[in_sess][~correct[in_sess]]]
            top = max(set(wrong_as), key=wrong_as.count) if wrong_as else '-'
            bar = '#' * int(round(acc * 20))
            print(f'    {session[-4:]}  {acc:.2f} {bar:<20}  mostly wrong as: {top}')

        # --- per position in the recording ---
        print('  by position within the recording:')
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = rows[(position[rows] >= lo) & (position[rows] < hi)]
            if len(band):
                print(f'    {lo:.1f}-{hi:.1f} of the way through: {correct[band].mean():.3f}  ({len(band)} frames)')

        # --- brightness of right vs wrong ---
        right_b, wrong_b = brightness[rows][ok], brightness[rows][~ok]
        print(f'  mean brightness: correct {right_b.mean():.1f}, wrong {wrong_b.mean():.1f}')

        # --- contact sheets ---
        wrong_idx = test[rows][~ok][:24]
        right_idx = test[rows][ok][:24]
        cv2.imwrite(str(args.out_dir / f'{name}_wrong.png'), tile(raw[wrong_idx])[:, :, ::-1])
        cv2.imwrite(str(args.out_dir / f'{name}_right.png'), tile(raw[right_idx])[:, :, ::-1])
        print(f'  sheets -> {args.out_dir}/{name}_wrong.png ({len(wrong_idx)}), {name}_right.png ({len(right_idx)})\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
