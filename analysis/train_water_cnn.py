#!/usr/bin/env python3
"""
Train and compare the four water-level classifiers: CNN encoders + MLP head.

    vision + tactile + audio      vision + tactile      vision + audio      vision only

Each is the same WaterClassifier with different encoders attached, trained on identical splits
with identical hyperparameters, so a gap between them is about the sensor.

HOW IT IS SCORED, and why not just training accuracy. At ~60 trials against 0.3-0.8 M parameters
every variant reaches 100% on its training set within a few epochs, including on a modality that
carries no information at all. So the reported number is held-out accuracy under stratified
K-fold: every trial is predicted exactly once, by a model that never saw it. Training accuracy is
printed too, but only to show that the fit succeeded -- it cannot rank the variants.

The shuffled-label control is not optional here. With 60 trials, a model this size scores well
above the nominal 1/3 on shuffled labels often enough that an unremarkable real result can look
like a finding. The run ends by measuring that floor on the richest modality set and printing the
threshold a result has to clear.

Usage (on lamb, ROS sourced for torch):
    python3 train_water_cnn.py --tensors water_tensors.npz --epochs 40
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from water_model import WaterClassifier  # noqa: E402

COMBOS = {
    'vision+tactile+audio': ('vision', 'tactile', 'audio'),
    'vision+tactile': ('vision', 'tactile'),
    'vision+audio': ('vision', 'audio'),
    'vision': ('vision',),
}


def load(path: pathlib.Path, device: str):
    """Load the npz into normalised float tensors on `device`."""
    d = np.load(path, allow_pickle=True)
    out = {
        'vision': torch.from_numpy(d['vision']).float().div_(255.0),
        'tactile': torch.from_numpy(d['tactile']).float().div_(255.0),
        'audio': torch.from_numpy(d['audio']).float(),
    }
    # Per-dataset standardisation of the log-mel: it is log-power, so its offset is arbitrary and
    # leaving it un-centred just makes the first conv layer spend capacity on a constant.
    a = out['audio']
    out['audio'] = (a - a.mean()) / (a.std() + 1e-6)
    y = torch.from_numpy(d['label']).long()
    return {k: v.to(device) for k, v in out.items()}, y.to(device), [str(s) for s in d['levels']]


def train_one(combo, X, y, tr_idx, va_idx, args, device):
    """
    Train one variant on `tr_idx`, evaluate on `va_idx`, return (val_acc, train_acc, preds).

    Mini-batched, not full-batch: at 224x224 with K=8 frames a single step over 75 trials pushes
    600 frames through the stem at once, and the first layer's activations alone run to ~1 GB.
    Batching by trial keeps it bounded and, at this size, costs nothing in wall time.
    """
    torch.manual_seed(args.seed)
    model = WaterClassifier(combo).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()

    def batches(idx, shuffle):
        order = torch.randperm(len(idx), device=device) if shuffle else torch.arange(len(idx), device=device)
        for i in range(0, len(idx), args.batch_size):
            sel = idx[order[i:i + args.batch_size]]
            yield {m: X[m][sel] for m in combo}, y[sel]

    for _ in range(args.epochs):
        model.train()
        for xb, yb in batches(tr_idx, shuffle=True):
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        def acc_and_pred(idx):
            preds = []
            for xb, _ in batches(idx, shuffle=False):
                preds.append(model(xb).argmax(1))
            pred = torch.cat(preds)
            return (pred == y[idx]).float().mean().item(), pred.cpu().numpy()

        train_acc, _ = acc_and_pred(tr_idx)
        val_acc, val_pred = acc_and_pred(va_idx)
    return val_acc, train_acc, val_pred



def main() -> int:
    """Evaluate every modality combination on the held-out trials, then the shuffled-label floor."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tensors', required=True, type=pathlib.Path)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-shuffles', type=int, default=3)
    ap.add_argument('--combos', nargs='+', default=list(COMBOS),
                    choices=list(COMBOS),
                    help='which modality combinations to train; default is all four')
    ap.add_argument('--skip-null', action='store_true',
                    help='skip the shuffled-label floor (it is the slowest part and only '
                         'meaningful once, so skip it when re-running a single variant)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from sklearn.metrics import confusion_matrix

    X, y, levels = load(args.tensors, args.device)
    d = np.load(args.tensors, allow_pickle=True)
    split = np.asarray([str(s) for s in d['split']])
    tr_idx = torch.as_tensor(np.flatnonzero(split == 'train'), device=args.device)
    va_idx = torch.as_tensor(np.flatnonzero(split == 'val'), device=args.device)
    y_va = y[va_idx].cpu().numpy()

    per_class = {lv: int((y == i).sum()) for i, lv in enumerate(levels)}
    print(f'trials: {len(y)}   per class: {per_class}   nominal chance {1 / len(levels):.3f}')
    print(f'split:  {len(tr_idx)} train / {len(va_idx)} val')
    print(f'device: {args.device}   {args.epochs} epochs, lr {args.lr}, batch {args.batch_size}\n')
    if len(va_idx) == 0:
        print('no validation trials -- was --val-from set too high at extraction?', file=sys.stderr)
        return 1

    results = {}
    selected = {k: COMBOS[k] for k in args.combos}
    for name, combo in selected.items():
        val_acc, train_acc, pred = train_one(combo, X, y, tr_idx, va_idx, args, args.device)
        results[name] = (val_acc, train_acc, confusion_matrix(y_va, pred, labels=range(len(levels))))
        params = WaterClassifier(combo).n_parameters()
        print(f'{name:<24} params={params / 1e6:.2f}M  train={train_acc:.3f}  val={val_acc:.3f}')

    print('\nconfusion matrices (rows = true, cols = predicted; validation trials only)')
    for name, (_, _, cm) in results.items():
        print(f'\n  {name}')
        print('         ' + ''.join(f'{lv:>8}' for lv in levels))
        for i, lv in enumerate(levels):
            print(f'  {lv:>6} ' + ''.join(f'{v:>8}' for v in cm[i]))

    # With only a handful of validation trials per class, a model that knows nothing still lands
    # well above 1/3 some of the time. This measures how high, using the richest modality set --
    # the one most able to memorise a shuffle -- so a real result has a bar to clear.
    if args.skip_null:
        print('\n(shuffled-label floor skipped; re-run without --skip-null for the real '
              'chance threshold)')
        null = np.array([1.0 / len(levels)])
    else:
        print(f'\nshuffled-label floor ({args.n_shuffles} runs, all three modalities):')
        rng = np.random.default_rng(args.seed)
        null = []
        for _ in range(args.n_shuffles):
            y_shuf = y.clone()
            y_shuf[tr_idx] = torch.as_tensor(
                rng.permutation(y[tr_idx].cpu().numpy()), device=args.device)
            acc, _, _ = train_one(('vision', 'tactile', 'audio'), X, y_shuf, tr_idx, va_idx,
                                  args, args.device)
            null.append(acc)
        null = np.asarray(null)
        print(f'  mean {null.mean():.3f}  max {null.max():.3f}')
        print(f'  => treat val <= {null.max():.3f} as indistinguishable from no signal')

    print('\nranking (validation):')
    for name, (acc, tr, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        mark = '' if acc > null.max() else '   (not above the shuffled floor)'
        print(f'  {acc:.3f}  {name:<24} (train {tr:.3f}){mark}')
    base = results['vision'][0] if 'vision' in results else float('nan')
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f'\nvision-only baseline: {base:.3f}')
    print(f'best: {best[0]} at {best[1][0]:.3f}  -> extra sensors worth {best[1][0] - base:+.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
