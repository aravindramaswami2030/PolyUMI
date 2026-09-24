#!/usr/bin/env python3
"""
Turn recorded PolyUMI scenes into a tactile texture-classification dataset (one .npz).

Input is the pzarr a scene already carries -- ``<scene>/scene.zarr/episode_N/finger/frames``,
the Pi's finger camera at its native 1152x648 -- and a mapping that says which scene holds which
texture. Output is one npz of uint8 images plus labels and a train/val/test assignment, which
``train_texture_cnn.py`` trains on with no zarr dependency (lamb has torch but not zarr; this
laptop's uv venv has zarr but not torch, so the two stages deliberately do not share an env).

**One sample is one frame.** A texture is visible in a single contact image, so there is nothing
to gain from stacking frames, and single frames make the set big enough to train a small CNN in
seconds.

**Splits are by EPISODE, never by frame.** Consecutive frames of an episode are near-duplicates:
same object, same grasp, same lighting, milliseconds apart. Split those at random and the val set
is a copy of the training set, which scores ~99% while proving nothing. Holding out whole
episodes asks the real question -- a texture seen in a grasp the model never saw. The split is
drawn per class so every class contributes to all three sets.

**Classes are balanced by construction**: after sampling, every class is truncated to the
smallest class's count in each split, so accuracy is directly comparable to 1/n_classes chance.

The crop is the `finger_rgb` contract from ingest/config/finger_camera.yaml (x_min=170 drops the
strip the gripper mount occludes), so a classifier cannot score on the mount instead of the
object.

Usage (laptop, uv venv -- it has zarr + imagecodecs):

    uv run python analysis/texture_dataset.py \
        --classes analysis/texture_classes.json \
        --recordings recordings --out texture_tensors.npz

where the mapping names one or more scenes per class:

    {"rough_wood": ["scene_2026-09-15_01-20-37_7006"], "smooth_metal": ["scene_..."]}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import zarr

# Registers the imagecodecs_jpegxl codec the finger frames are compressed with. Importing the
# project's own store module rather than re-registering by hand keeps one registration site.
from polyumi_ingest.camera_preproc import crop_finger_rgb
from polyumi_ingest.pzarr import store as _pzarr_store  # noqa: F401 - registers Jpegxl

#: Square side the frames are resized to after cropping. 128 rather than the policies' 224: this
#: is a texture classifier, not a policy input, and the smaller image is ~3x faster to train on
#: while leaving the surface detail a texture actually lives in.
IMG = 128

#: Frames taken per episode, evenly spaced across it. Even spacing rather than a window because
#: the grasp happens at a different moment in every episode; spreading the samples means no class
#: is represented only by the approach or only by the release.
FRAMES_PER_EPISODE = 24

#: The finger_rgb crop contract (ingest/config/finger_camera.yaml).
CROP_X_MIN = 170


def load_mapping(path: pathlib.Path) -> dict[str, list[str]]:
    """
    Read ``{"class": ["scene_dir", ...]}``, rejecting empty classes.

    Several scenes may share a class -- that is how a texture recorded over two sessions stays one
    label -- but a scene must not appear under two classes, which would make the label ambiguous.
    """
    raw = json.loads(pathlib.Path(path).read_text())
    mapping = {k: list(v) for k, v in raw.items() if not k.startswith('_')}
    if not mapping:
        raise ValueError(f'{path}: no classes found')
    seen: dict[str, str] = {}
    for label, scenes in mapping.items():
        if not scenes:
            raise ValueError(f'{path}: class {label!r} lists no scenes')
        for scene in scenes:
            if scene in seen:
                raise ValueError(f'{path}: scene {scene!r} is listed under both {seen[scene]!r} and {label!r}')
            seen[scene] = label
    return mapping


def episode_frame_counts(scene_zarr: pathlib.Path) -> dict[str, int]:
    """Return ``{episode name: frame count}`` for every episode in a scene that has finger frames."""
    root = zarr.open_group(str(scene_zarr), mode='r', zarr_format=2)
    counts = {}
    for name in sorted(root.group_keys(), key=lambda n: int(n.rsplit('_', 1)[1]) if n[-1].isdigit() else 0):
        try:
            counts[name] = int(root[f'{name}/finger/frames'].shape[0])
        except KeyError:
            continue  # a MAPPING session, or an episode whose finger stream never arrived
    return counts


def sample_indices(n_frames: int, k: int) -> np.ndarray:
    """Evenly spaced frame indices across an episode, without repeats when the episode is short."""
    if n_frames <= 0:
        return np.empty(0, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, min(k, n_frames)).round().astype(int))


def read_frames(scene_zarr: pathlib.Path, episode: str, indices: np.ndarray, img: int) -> np.ndarray:
    """
    Decode the named frames, crop away the mount, and resize to ``img`` x ``img``.

    Only the requested frames are decoded -- the array is chunked one frame per chunk, so this
    costs the sampled frames and not the whole episode.
    """
    root = zarr.open_group(str(scene_zarr), mode='r', zarr_format=2)
    frames = root[f'{episode}/finger/frames']
    out = np.empty((len(indices), img, img, 3), dtype=np.uint8)
    for i, index in enumerate(indices):
        cropped = crop_finger_rgb(np.asarray(frames[int(index)]), x_min=CROP_X_MIN)
        out[i] = cv2.resize(cropped, (img, img), interpolation=cv2.INTER_AREA)
    return out


def assign_episode_splits(
    episodes: list[tuple[str, str, int]],
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> dict[tuple[str, str], str]:
    """
    Split one class's episodes into train/val/test, by FRAME count rather than episode count.

    Episodes differ in length by several times, so dealing them out by count would put a wildly
    different number of frames in each split. Episodes are shuffled, then taken into val and test
    until each has its share of the class's frames -- whole episodes only, so no frame of an
    episode ever lands on the other side of a split boundary.

    Episodes are taken while doing so moves the split's running total CLOSER to its target, rather
    than until the target is passed: whole episodes are a coarse unit, so stopping at the first
    total over the line systematically overshoots (a 53-frame target built from 24-frame episodes
    lands on 72 rather than 48). The realised split still cannot be exact -- it is quantised by
    episode length -- but it is the closest reachable rather than the first one past.

    At least one episode goes to val and one to test whenever there are three or more; with fewer
    than three the caller is told, because a class cannot be held out at all in that case.
    """
    order = list(episodes)
    rng.shuffle(order)
    total = sum(n for _, _, n in order)
    want = {'test': total * test_frac, 'val': total * val_frac}
    assigned: dict[tuple[str, str], str] = {}
    for split in ('test', 'val'):
        taken = 0
        # Leave an episode for val as well as for train while filling test, and one for train
        # while filling val -- a split that swallowed them all would leave nothing to fit on.
        min_left = 2 if split == 'test' else 1
        for scene, episode, n in order:
            if (scene, episode) in assigned:
                continue
            if sum(1 for s, e, _ in order if (s, e) not in assigned) <= min_left:
                break
            if taken > 0 and abs(taken + n - want[split]) >= abs(taken - want[split]):
                break
            assigned[(scene, episode)] = split
            taken += n
    for scene, episode, _ in order:
        assigned.setdefault((scene, episode), 'train')
    return assigned


def balance(labels: np.ndarray, splits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return the indices keeping every class the same size within each split.

    Truncating to the smallest class is the honest option here: with unequal classes, accuracy
    rewards a model for learning the prior rather than the texture, and the confusion matrix stops
    being readable row by row. Which frames are dropped is random, not the tail, so no episode is
    systematically preferred.
    """
    keep: list[np.ndarray] = []
    for split in ('train', 'val', 'test'):
        in_split = np.flatnonzero(splits == split)
        if in_split.size == 0:
            continue
        per_class = [in_split[labels[in_split] == c] for c in np.unique(labels)]
        smallest = min(len(idx) for idx in per_class)
        for idx in per_class:
            keep.append(rng.permutation(idx)[:smallest])
    return np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=int)


def main() -> int:
    """Build the npz from the scene mapping."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--classes', type=pathlib.Path, required=True, help='JSON: {"class": ["scene_dir", ...]}')
    ap.add_argument('--recordings', type=pathlib.Path, default=pathlib.Path('recordings'))
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('texture_tensors.npz'))
    ap.add_argument('--frames-per-episode', type=int, default=FRAMES_PER_EPISODE)
    ap.add_argument('--img', type=int, default=IMG)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--no-balance',
        action='store_true',
        help='keep every sampled frame instead of truncating each class to the smallest',
    )
    args = ap.parse_args()

    mapping = load_mapping(args.classes)
    rng = np.random.default_rng(args.seed)
    class_names = sorted(mapping)

    images: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    scenes_of: list[str] = []
    episodes_of: list[str] = []

    for label_index, label in enumerate(class_names):
        per_class: list[tuple[str, str, int]] = []
        raw_counts: dict[tuple[str, str], int] = {}
        for scene_name in mapping[label]:
            scene_zarr = args.recordings / scene_name / 'scene.zarr'
            if not scene_zarr.is_dir():
                print(f'error: no scene.zarr at {scene_zarr}', file=sys.stderr)
                return 1
            for episode, n_frames in episode_frame_counts(scene_zarr).items():
                # Weight each episode by the frames that will actually land in the npz, not by its
                # full length: only --frames-per-episode of them are kept, and episodes differ in
                # length several-fold, so targeting raw lengths drifts the realised split away
                # from --val-frac / --test-frac.
                raw_counts[(scene_name, episode)] = n_frames
                per_class.append((scene_name, episode, len(sample_indices(n_frames, args.frames_per_episode))))
        if len(per_class) < 3:
            print(
                f'error: class {label!r} has {len(per_class)} episode(s) with finger frames; '
                'at least 3 are needed to hold out a val and a test episode.',
                file=sys.stderr,
            )
            return 1

        assignment = assign_episode_splits(per_class, args.val_frac, args.test_frac, rng)
        for scene_name, episode, _ in per_class:
            indices = sample_indices(raw_counts[(scene_name, episode)], args.frames_per_episode)
            frames = read_frames(args.recordings / scene_name / 'scene.zarr', episode, indices, args.img)
            images.append(frames)
            labels.extend([label_index] * len(frames))
            splits.extend([assignment[(scene_name, episode)]] * len(frames))
            scenes_of.extend([scene_name] * len(frames))
            episodes_of.extend([episode] * len(frames))
        sampled = sum(n for _, _, n in per_class)
        available = sum(raw_counts.values())
        print(f'{label:>20}: {len(per_class)} episodes, {sampled} frames sampled of {available} available')

    x = np.concatenate(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    split = np.asarray(splits)
    scene_ids = np.asarray(scenes_of)
    episode_ids = np.asarray(episodes_of)

    if not args.no_balance:
        keep = balance(y, split, rng)
        x, y, split, scene_ids, episode_ids = (a[keep] for a in (x, y, split, scene_ids, episode_ids))

    np.savez_compressed(
        args.out,
        images=x,
        labels=y,
        split=split,
        classes=np.asarray(class_names),
        scene=scene_ids,
        episode=episode_ids,
    )

    print(f'\nwrote {args.out}  ({x.nbytes / 1e6:.0f} MB uncompressed, {x.shape[0]} frames of {args.img}x{args.img})')
    for name in ('train', 'val', 'test'):
        rows = split == name
        counts = {class_names[c]: int(((y == c) & rows).sum()) for c in range(len(class_names))}
        n_eps = len({(s, e) for s, e in zip(scene_ids[rows], episode_ids[rows])})
        print(f'  {name:>5}: {int(rows.sum()):>6} frames from {n_eps:>3} episodes   {counts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
