#!/usr/bin/env python3
"""
Cross-validate the texture classifier over the training pool, then refit and score the test set.

Two stages, matching how the water classifier is scored:

1. **K-fold cross-validation over the pool.** The pool's SESSIONS are dealt into K folds, stratified
   by class; each fold trains a fresh model on the rest and is scored on itself. Every pool frame
   is therefore predicted exactly once, by a model that never saw its session. This is the number
   that says whether the setup generalises, and the fold spread says how much to trust it.
2. **A final refit on the whole pool**, trained for the full epoch budget, scored once on the
   held-out test sessions.

**Folds are over sessions, never frames.** Frames of one session are the same grasp a tenth of a
second apart; a frame-level fold puts near-duplicates on both sides and reports ~100% regardless
of whether the model learned anything.

Train accuracy is printed only to show the fit succeeded. At this capacity it reaches 100% on
anything, so it ranks nothing.

Usage (laptop ROS python has torch + the GPU):

    /usr/bin/python3 analysis/train_texture_kfold.py --tensors texture_ranges.npz --epochs 120
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_texture_cnn import (  # noqa: E402
    TextureClassifier,
    confusion,
    per_class_accuracy,
)


def load(path: pathlib.Path, device: str):
    """Load the npz: images NCHW in [0, 1], labels, split names, session ids, class names."""
    d = np.load(path, allow_pickle=False)
    x = torch.from_numpy(d['images']).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y = torch.from_numpy(d['labels']).long()
    return (
        x.to(device),
        y.to(device),
        np.asarray(d['split']),
        np.asarray(d['episode']),
        [str(c) for c in d['classes']],
    )


def session_folds(sessions: np.ndarray, labels: np.ndarray, pool: np.ndarray, k: int, rng) -> list[np.ndarray]:
    """
    Deal the pool's sessions into `k` folds, stratified by class, and return frame indices per fold.

    Dealing each class's shuffled sessions round-robin keeps every fold class-balanced, so a fold's
    accuracy is comparable to 1/n_classes without reweighting.
    """
    fold_of: dict[str, int] = {}
    for label in np.unique(labels[pool]):
        class_sessions = sorted(set(sessions[pool][labels[pool] == label]))
        rng.shuffle(class_sessions)
        for i, session in enumerate(class_sessions):
            fold_of[session] = i % k
    return [pool[np.array([fold_of[s] for s in sessions[pool]]) == f] for f in range(k)]


def train_model(x, y, train_idx, val_idx, args, device, n_classes: int, seed: int):
    """
    Train one model. With `val_idx`, the best-val-loss epoch is restored; without, it runs straight.

    The final refit deliberately has no validation set -- every pool session is training data by
    then -- so it trains the full budget rather than early-stopping on something it also scores.
    """
    torch.manual_seed(seed)
    model = TextureClassifier(n_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def batches(indices, shuffle):
        order = indices[torch.randperm(len(indices), device=device)] if shuffle else indices
        for start in range(0, len(order), args.batch):
            chunk = order[start : start + args.batch]
            yield x[chunk], y[chunk]

    best_loss, best_state, best_epoch = float('inf'), None, args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in batches(train_idx, shuffle=True):
            if args.augment and torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[-1])
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        if val_idx is not None and len(val_idx):
            model.eval()
            with torch.no_grad():
                total = sum(loss_fn(model(xb), yb).item() * len(yb) for xb, yb in batches(val_idx, shuffle=False))
            val_loss = total / len(val_idx)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss


@torch.no_grad()
def predict(model, x, indices, batch: int) -> np.ndarray:
    """Predicted labels for `indices`."""
    model.eval()
    preds = [model(x[indices[s : s + batch]]).argmax(1) for s in range(0, len(indices), batch)]
    return torch.cat(preds).cpu().numpy() if preds else np.empty(0, dtype=int)


def main() -> int:
    """Cross-validate, refit, report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tensors', type=pathlib.Path, default=pathlib.Path('texture_ranges.npz'))
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, default=1, help='repeat the whole CV + refit this many times and average')
    ap.add_argument('--no-augment', dest='augment', action='store_false')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    x, y, split, sessions, classes = load(args.tensors, args.device)
    y_np = y.cpu().numpy()
    pool = np.flatnonzero(split == 'pool')
    test = np.flatnonzero(split == 'test')
    k = len(classes)

    print(f'classes: {classes}   chance {1 / k:.3f}')
    print(f'pool:    {len(pool)} frames from {len(set(sessions[pool]))} sessions')
    print(f'test:    {len(test)} frames from {len(set(sessions[test]))} sessions')
    print(f'device:  {args.device}   {args.folds}-fold CV then a {args.epochs}-epoch refit\n')

    cv_accs, test_accs, per_class, cms = [], [], [], []
    for s in range(args.seeds):
        # Seeds are spaced so a run's fold seeds (seed + fold index) cannot collide with the next
        # run's, which would otherwise make two "independent" repeats share a trained model.
        seed = args.seed + 100 * s

        # --- 1. cross-validation over the pool ---------------------------------------------
        rng = np.random.default_rng(seed)
        folds = session_folds(sessions, y_np, pool, args.folds, rng)
        cv_pred = np.zeros(len(y_np), dtype=int)
        for f, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(pool, val_idx)
            model, _, _ = train_model(
                x,
                y,
                torch.as_tensor(train_idx, device=args.device),
                torch.as_tensor(val_idx, device=args.device),
                args,
                args.device,
                k,
                seed=seed + f,
            )
            cv_pred[val_idx] = predict(model, x, torch.as_tensor(val_idx, device=args.device), args.batch)
        cv_acc = float((cv_pred[pool] == y_np[pool]).mean())

        # --- 2. refit on the whole pool, score the held-out test ----------------------------
        model, _, _ = train_model(
            x, y, torch.as_tensor(pool, device=args.device), None, args, args.device, k, seed=seed
        )
        test_pred = predict(model, x, torch.as_tensor(test, device=args.device), args.batch)
        test_acc = float((test_pred == y_np[test]).mean())
        cm = confusion(y_np[test], test_pred, k)

        cv_accs.append(cv_acc)
        test_accs.append(test_acc)
        cms.append(cm)
        per_class.append(per_class_accuracy(cm))
        print(f'  seed {s + 1}/{args.seeds}: cross-val {cv_acc:.3f}   test {test_acc:.3f}')

    total_cm = np.sum(cms, axis=0)
    pc = np.stack(per_class)

    print('\n=== per-class TEST accuracy ' + '=' * 37)
    print(f'  {"class":>12}{"accuracy":>12}{"sd":>8}')
    for i, name in enumerate(classes):
        print(f'  {name:>12}{pc[:, i].mean():>12.3f}{pc[:, i].std():>8.3f}')

    # Each cell is a decimal share of its own row, so a row sums to 1.000 and the diagonal is the
    # per-class accuracy printed just above -- the two tables then read as one. Counts are summed
    # across seeds before normalising, which equals averaging the seeds' rows here because every
    # seed scores the same frames.
    print('\n=== TEST confusion (rows = true, cols = predicted, decimal of row total) ' + '=' * 2)
    width = max(12, max(len(c) for c in classes) + 2)
    print(' ' * (width + 2) + ''.join(f'{c:>{width}}' for c in classes))
    for i, name in enumerate(classes):
        row = total_cm[i] / max(int(total_cm[i].sum()), 1)
        print(f'  {name:>{width}}' + ''.join(f'{v:>{width}.3f}' for v in row))

    print(
        f'\noverall test accuracy {np.mean(test_accs):.3f} +- {np.std(test_accs):.3f} '
        f'over {args.seeds} seed(s), chance {1 / k:.3f}.'
    )
    print(
        f'(the {args.folds}-fold cross-validated estimate was '
        f'{np.mean(cv_accs):.3f} +- {np.std(cv_accs):.3f}, for reference)'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
