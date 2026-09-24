#!/usr/bin/env python3
"""
Train the tactile texture classifier and report accuracy, per-class accuracy and confusion.

The encoder is ``water_model.ImageEncoder`` -- the same four stride-2 conv/BN/ReLU blocks the
water classifier's tactile branch uses -- with a small dropout head on top. Reusing it rather than
writing a new one means a result here is comparable with the water numbers, and the architecture
is already known to train on a few thousand images in seconds.

It trains in well under a minute on a laptop GPU: 128x128 frames, batch 64, and an encoder of
~0.3 M parameters. Measured: 1000 frames, 40 epochs, 11 s end to end on an RTX 4060.

**What the numbers mean.** The dataset holds out whole EPISODES (see texture_dataset.py), so val
and test scores answer "a texture in a grasp the model never saw", not "a frame next to one it
memorised". Classes are balanced, so chance is 1/n_classes and each confusion-matrix row has the
same total. Train accuracy is printed only to show the fit succeeded -- at this capacity it
reaches ~100% on anything, including noise, so it cannot rank anything.

The best-val-loss epoch is restored before scoring, so the reported val and test numbers come
from the same checkpoint you would keep.

Usage (laptop ROS python has torch + the GPU; lamb works too):

    /usr/bin/python3 analysis/train_texture_cnn.py --tensors texture_tensors.npz
    /usr/bin/python3 analysis/train_texture_cnn.py --tensors texture_tensors.npz --seeds 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from water_model import ImageEncoder  # noqa: E402 - after the sys.path insert that finds it


class TextureClassifier(nn.Module):
    """The water classifier's tactile encoder plus a dropout head, over single frames."""

    def __init__(self, n_classes: int, d_embed: int = 128, hidden: int = 256, dropout: float = 0.5):
        """Build the shared-shape encoder and a head sized for `n_classes`."""
        super().__init__()
        self.encoder = ImageEncoder(d_embed=d_embed)
        self.head = nn.Sequential(
            nn.Linear(d_embed, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, n_classes). The encoder's frame axis is added and removed here."""
        return self.head(self.encoder(x.unsqueeze(1)))

    def n_parameters(self) -> int:
        """Total trainable parameters, for the capacity note in the report."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load(path: pathlib.Path, device: str):
    """Load the npz into a (images, labels, split, class names) tuple, images NCHW float in [0, 1]."""
    d = np.load(path, allow_pickle=False)
    x = torch.from_numpy(d['images']).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y = torch.from_numpy(d['labels']).long()
    return x.to(device), y.to(device), np.asarray(d['split']), [str(c) for c in d['classes']]


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Return counts with rows = true, cols = predicted. Hand-rolled: sklearn is absent in these envs."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall per class: the diagonal over each row's total, NaN for a class with no samples."""
    totals = cm.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(totals > 0, np.diag(cm) / np.maximum(totals, 1), np.nan)


def train_once(x, y, idx, args, device, n_classes: int, seed: int):
    """
    Train one model on `idx['train']`, restoring the best-val-loss epoch. Returns (model, history).

    Early stopping is on val LOSS rather than val accuracy: accuracy over a few hundred held-out
    frames moves in visible steps and picks a checkpoint by luck of the rounding, while the loss
    keeps ranking checkpoints that score the same.
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

    best_loss, best_state, best_epoch = float('inf'), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in batches(idx['train'], shuffle=True):
            # Flip augmentation only: a texture is the same texture mirrored, but it is NOT the
            # same rotated onto a different part of the finger, and colour jitter would erase the
            # surface tone that distinguishes several of these materials.
            if args.augment and torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[-1])
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            losses = [loss_fn(model(xb), yb).item() * len(yb) for xb, yb in batches(idx['val'], shuffle=False)]
        val_loss = sum(losses) / max(len(idx['val']), 1)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_val_loss': best_loss}


@torch.no_grad()
def predict(model, x, y, indices, args) -> tuple[np.ndarray, np.ndarray]:
    """Return (true, predicted) label arrays for `indices`."""
    model.eval()
    preds = []
    for start in range(0, len(indices), args.batch):
        chunk = indices[start : start + args.batch]
        preds.append(model(x[chunk]).argmax(1))
    pred = torch.cat(preds).cpu().numpy() if preds else np.empty(0, dtype=int)
    return y[indices].cpu().numpy(), pred


def show_matrix(cm: np.ndarray, classes: list[str], title: str) -> None:
    """Print one confusion matrix, rows = true, cols = predicted."""
    width = max(12, max(len(c) for c in classes) + 2)
    print(f'\n  {title}')
    print(' ' * (width + 2) + ''.join(f'{c:>{width}}' for c in classes))
    for i, name in enumerate(classes):
        print(f'  {name:>{width}}' + ''.join(f'{v:>{width}}' for v in cm[i]))


def main() -> int:
    """Train, score, and report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tensors', type=pathlib.Path, default=pathlib.Path('texture_tensors.npz'))
    # 40 epochs at batch 64 rather than fewer, larger steps: a class of ~20 episodes yields only a
    # few hundred training frames, and at batch 128 that is two optimizer steps per epoch -- too
    # few to leave the initialisation, which shows up as a model that predicts one class.
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--seeds', type=int, default=1, help='train this many models and average the scores')
    ap.add_argument('--no-augment', dest='augment', action='store_false', help='disable horizontal flips')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    x, y, split, classes = load(args.tensors, args.device)
    idx = {
        name: torch.as_tensor(np.flatnonzero(split == name), device=args.device) for name in ('train', 'val', 'test')
    }
    k = len(classes)
    counts = {name: len(indices) for name, indices in idx.items()}
    print(f'classes: {classes}   chance {1 / k:.3f}')
    print(f'frames:  {counts["train"]} train / {counts["val"]} val / {counts["test"]} test')
    print(f'device:  {args.device}   {args.epochs} epochs, batch {args.batch}, lr {args.lr}\n')
    if counts['val'] == 0 or counts['test'] == 0:
        print('error: the npz has an empty val or test split -- re-extract with more episodes.', file=sys.stderr)
        return 1

    accuracies = {'train': [], 'val': [], 'test': []}
    per_class = {'val': [], 'test': []}
    matrices = {'val': [], 'test': []}
    for seed in range(args.seeds):
        model, history = train_once(x, y, idx, args, args.device, k, seed)
        line = [
            f'seed {seed}: best epoch {history["best_epoch"]}/{args.epochs} (val loss {history["best_val_loss"]:.4f})'
        ]
        for name in ('train', 'val', 'test'):
            true, pred = predict(model, x, y, idx[name], args)
            accuracies[name].append(float((true == pred).mean()))
            if name in matrices:
                cm = confusion(true, pred, k)
                matrices[name].append(cm)
                per_class[name].append(per_class_accuracy(cm))
            line.append(f'{name} {accuracies[name][-1]:.3f}')
        print('  '.join(line))
        if seed == 0:
            print(f'         model: {model.n_parameters() / 1e6:.2f}M parameters')

    print('\n=== overall accuracy ' + '=' * 40)
    for name in ('train', 'val', 'test'):
        values = np.asarray(accuracies[name])
        spread = f' +- {values.std():.3f}' if args.seeds > 1 else ''
        print(f'  {name:>5}: {values.mean():.3f}{spread}')

    print('\n=== per-class accuracy (recall) ' + '=' * 29)
    print(f'  {"class":>20}{"val":>12}{"test":>12}')
    val_pc = np.nanmean(np.stack(per_class['val']), axis=0)
    test_pc = np.nanmean(np.stack(per_class['test']), axis=0)
    for i, name in enumerate(classes):
        print(f'  {name:>20}{val_pc[i]:>12.3f}{test_pc[i]:>12.3f}')

    print('\n=== confusion matrices (rows = true, cols = predicted) ' + '=' * 6)
    if args.seeds > 1:
        print(f'  summed over {args.seeds} seeds')
    for name in ('val', 'test'):
        show_matrix(np.sum(matrices[name], axis=0), classes, f'{name} split')

    print(f'\nchance is {1 / k:.3f}; classes are balanced, so every row of a matrix has the same total.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
