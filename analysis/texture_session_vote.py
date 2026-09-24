#!/usr/bin/env python3
"""
Score the texture classifier per RECORDING rather than per frame.

The model classifies one frame at a time, so the headline accuracy is a per-frame figure over
frames that are not independent -- 20 of them come from each recording. This aggregates a
recording's frame predictions into a single decision, which is the quantity that answers "can it
identify the shape from a grasp" and whose sample size (35 held-out recordings) is the honest one.

Two aggregations, because they disagree when a recording is split:

* majority vote -- each frame gets one vote, ties broken toward the lowest class index.
* mean probability -- softmax averaged over the recording's frames, then argmax. This lets a
  confident frame outweigh several hesitant ones, which is usually what you want when half a
  recording shows no contact at all.

The model, the split and the training budget are exactly those of train_texture_kfold.py's final
refit; only the scoring changes.

Usage:
    /usr/bin/python3 analysis/texture_session_vote.py --tensors texture_full.npz --seeds 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_texture_cnn import confusion, per_class_accuracy  # noqa: E402
from train_texture_kfold import load, train_model  # noqa: E402


@torch.no_grad()
def probabilities(model, x, indices, batch: int) -> np.ndarray:
    """Softmax probabilities for `indices`, as (n, n_classes)."""
    model.eval()
    out = [model(x[indices[s : s + batch]]).softmax(-1) for s in range(0, len(indices), batch)]
    return torch.cat(out).cpu().numpy()


def main() -> int:
    """Refit, then score per frame and per recording."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tensors', type=pathlib.Path, default=pathlib.Path('texture_full.npz'))
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-augment', dest='augment', action='store_false')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    x, y, split, sessions, classes = load(args.tensors, args.device)
    y_np = y.cpu().numpy()
    pool = np.flatnonzero(split == 'pool')
    test = np.flatnonzero(split == 'test')
    k = len(classes)
    test_sessions = sorted(set(sessions[test]))

    print(f'classes: {classes}   chance {1 / k:.3f}')
    print(f'test: {len(test)} frames from {len(test_sessions)} recordings ({len(test_sessions) // k} per class)\n')

    frame_accs, vote_accs, prob_accs = [], [], []
    vote_cms, prob_cms, prob_pc = [], [], []
    for s in range(args.seeds):
        seed = args.seed + 100 * s
        model, _, _ = train_model(
            x, y, torch.as_tensor(pool, device=args.device), None, args, args.device, k, seed=seed
        )
        probs = probabilities(model, x, torch.as_tensor(test, device=args.device), args.batch)
        frame_pred = probs.argmax(1)
        frame_accs.append(float((frame_pred == y_np[test]).mean()))

        truth, vote_pred, prob_pred = [], [], []
        for session in test_sessions:
            rows = np.flatnonzero(sessions[test] == session)
            truth.append(y_np[test][rows][0])
            vote_pred.append(int(np.bincount(frame_pred[rows], minlength=k).argmax()))
            prob_pred.append(int(probs[rows].mean(axis=0).argmax()))
        truth = np.asarray(truth)
        vote_pred, prob_pred = np.asarray(vote_pred), np.asarray(prob_pred)

        vote_accs.append(float((vote_pred == truth).mean()))
        prob_accs.append(float((prob_pred == truth).mean()))
        vote_cms.append(confusion(truth, vote_pred, k))
        prob_cms.append(confusion(truth, prob_pred, k))
        prob_pc.append(per_class_accuracy(prob_cms[-1]))
        print(
            f'  seed {s + 1}/{args.seeds}: per-frame {frame_accs[-1]:.3f}   '
            f'per-recording vote {vote_accs[-1]:.3f}   mean-prob {prob_accs[-1]:.3f}'
        )

    print('\n=== overall accuracy ' + '=' * 44)
    print(f'  per frame   ({len(test)} frames):      {np.mean(frame_accs):.3f} +- {np.std(frame_accs):.3f}')
    print(f'  per recording, majority vote:     {np.mean(vote_accs):.3f} +- {np.std(vote_accs):.3f}')
    print(f'  per recording, mean probability:  {np.mean(prob_accs):.3f} +- {np.std(prob_accs):.3f}')

    pc = np.stack(prob_pc)
    print('\n=== per-class RECORDING accuracy (mean probability) ' + '=' * 13)
    print(f'  {"class":>12}{"accuracy":>12}{"sd":>8}   (out of {len(test_sessions) // k} recordings)')
    for i, name in enumerate(classes):
        print(f'  {name:>12}{pc[:, i].mean():>12.3f}{pc[:, i].std():>8.3f}')

    total = np.sum(prob_cms, axis=0)
    print('\n=== RECORDING confusion (rows = true, cols = predicted, decimal of row total) ' + '=' * 2)
    width = max(12, max(len(c) for c in classes) + 2)
    print(' ' * (width + 2) + ''.join(f'{c:>{width}}' for c in classes))
    for i, name in enumerate(classes):
        row = total[i] / max(int(total[i].sum()), 1)
        print(f'  {name:>{width}}' + ''.join(f'{v:>{width}.3f}' for v in row))
    print(
        f'\ncounts summed over {args.seeds} seeds, so each row totals {args.seeds * len(test_sessions) // k} decisions.'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
