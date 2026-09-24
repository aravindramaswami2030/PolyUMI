#!/usr/bin/env python3
"""
Turn a `texture_collect` image folder into the npz ``train_texture_cnn.py`` trains on.

``texture_collect`` writes one PNG per sample, laid out as ``<root>/<class>/<class>_<object>_<n>.png``
with a ``manifest.jsonl`` beside it. This reads that tree and emits the same npz the scene and
session extractors do, so the identical training command runs on collected data.

**One sample is one grasp**, not one video frame, and that changes what a split means. Frames
sampled from a recording are near-duplicates seconds apart, so they must be split by recording;
collected samples are separate grasps at randomised poses, so splitting them at random is honest.
The stricter option is still there: ``--split-by block`` holds out whole collection blocks, which
also separates the splits in time and across object swaps.

Classes are balanced by truncation, so chance is 1/n_classes and every confusion-matrix row has
the same total.

Usage:

    uv run python analysis/folder_texture_dataset.py --root /data/texture_dataset \
        --out texture_tensors.npz
    /usr/bin/python3 analysis/train_texture_cnn.py --tensors texture_tensors.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from texture_dataset import IMG, balance  # noqa: E402

#: Samples per class in one collection block, matching collect_texture_dataset.sh's BLOCK. Only
#: used by --split-by block, to recover which block a sample number belongs to.
BLOCK = 10


def parse_sample(path: pathlib.Path) -> tuple[str, int]:
    """
    Pull ``(object, index)`` out of a ``<class>_<object>_<n>.png`` filename.

    The object matters because a class collected from several objects can be checked afterwards
    for whether the model separated the class or memorised one object, which the manifest also
    records but the array layout would otherwise lose.
    """
    stem = path.stem
    label = path.parent.name
    rest = stem[len(label) + 1 :] if stem.startswith(f'{label}_') else stem
    obj, _, number = rest.rpartition('_')
    return (obj or 'unknown'), (int(number) if number.isdigit() else 0)


def main() -> int:
    """Build the npz from a collected image folder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=pathlib.Path, required=True, help='collection root, one directory per class')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('texture_tensors.npz'))
    ap.add_argument('--img', type=int, default=IMG)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument(
        '--split-by',
        choices=('sample', 'block'),
        default='sample',
        help='sample: random per grasp (default). block: hold out whole collection blocks.',
    )
    ap.add_argument('--block', type=int, default=BLOCK, help='samples per block, for --split-by block')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-balance', action='store_true')
    args = ap.parse_args()

    class_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    if not class_dirs:
        print(f'error: no class directories under {args.root}', file=sys.stderr)
        return 1
    class_names = [p.name for p in class_dirs]
    rng = np.random.default_rng(args.seed)

    images, labels, splits, objects, blocks = [], [], [], [], []
    for label_index, class_dir in enumerate(class_dirs):
        files = sorted(class_dir.glob('*.png'))
        if not files:
            print(f'error: {class_dir} holds no PNGs', file=sys.stderr)
            return 1

        parsed = [parse_sample(f) for f in files]
        block_of = [(index - 1) // args.block for _, index in parsed]

        if args.split_by == 'block':
            unique = sorted(set(block_of))
            rng.shuffle(unique)
            n_test = max(1, int(round(len(unique) * args.test_frac)))
            n_val = max(1, int(round(len(unique) * args.val_frac)))
            role = {b: 'test' for b in unique[:n_test]}
            role.update({b: 'val' for b in unique[n_test : n_test + n_val]})
            assignment = [role.get(b, 'train') for b in block_of]
        else:
            order = rng.permutation(len(files))
            n_test = max(1, int(round(len(files) * args.test_frac)))
            n_val = max(1, int(round(len(files) * args.val_frac)))
            assignment = ['train'] * len(files)
            for i in order[:n_test]:
                assignment[i] = 'test'
            for i in order[n_test : n_test + n_val]:
                assignment[i] = 'val'

        for path, (obj, _), block, split in zip(files, parsed, block_of, assignment):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f'error: could not decode {path}', file=sys.stderr)
                return 1
            images.append(cv2.resize(bgr[:, :, ::-1], (args.img, args.img), interpolation=cv2.INTER_AREA))
            labels.append(label_index)
            splits.append(split)
            objects.append(obj)
            blocks.append(f'block_{block}')
        print(f'{class_dir.name:>14}: {len(files)} samples, objects {sorted({o for o, _ in parsed})}')

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)
    split = np.asarray(splits)
    object_arr = np.asarray(objects)
    block_arr = np.asarray(blocks)

    if not args.no_balance:
        keep = balance(y, split, rng)
        x, y, split, object_arr, block_arr = (a[keep] for a in (x, y, split, object_arr, block_arr))

    np.savez_compressed(
        args.out,
        images=x,
        labels=y,
        split=split,
        classes=np.asarray(class_names),
        scene=object_arr,
        episode=block_arr,
    )

    print(f'\nwrote {args.out}  ({x.shape[0]} samples of {args.img}x{args.img}, split by {args.split_by})')
    for name in ('train', 'val', 'test'):
        rows = split == name
        counts = {class_names[c]: int(((y == c) & rows).sum()) for c in range(len(class_names))}
        print(f'  {name:>5}: {int(rows.sum()):>5}   {counts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
