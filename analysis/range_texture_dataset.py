#!/usr/bin/env python3
"""
Build a texture dataset from CONTIGUOUS RANGES of sessions in one scene.

A collection run records each class back to back, so a class is a slice of the scene's session
list rather than a set of names: "empty is everything from 1f5e through 7e7f". This takes those
boundary pairs, expands each into the sessions between them in recording order, and splits each
class into a training pool and a held-out test set at SESSION level.

**Sessions are addressed by position, not by name.** Four-character suffixes are not unique across
a long scene -- this corpus has ``1ebf`` twice, once in empty and once in square -- so a boundary
is matched to a session index and the range is the slice between the two indices. Resolving the
same string two ways is how a class quietly acquires another class's episodes.

**Splits are session-level and random within a class.** ``--test-sessions`` whole sessions per
class are drawn (seeded) as the held-out test set and the rest form the training pool; the pool is
folded by the trainer, also at session level. Frames from one session never straddle a split,
because at 10 fps they are near-duplicates and would make every score meaningless.

Classes end up with identical frame counts by construction: the same number of sessions and the
same number of frames sampled from each, so chance is 1/n_classes exactly.

Usage:

    uv run python analysis/range_texture_dataset.py \
        --scene recordings/scene_2026-09-16_04-24-12_72ea \
        --ranges analysis/texture_ranges.json \
        --out texture_ranges.npz --frames-per-session 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from texture_dataset import CROP_X_MIN, IMG, sample_indices  # noqa: E402

from polyumi_ingest.camera_preproc import crop_finger_rgb  # noqa: E402

#: Frames taken per session, evenly spaced. Sessions here run 14-144 frames, so a fixed count per
#: session keeps every session's contribution equal -- otherwise a long recording would dominate
#: its class and the model would learn that recording rather than the shape.
FRAMES_PER_SESSION = 20

#: Sessions per class held out entirely for the final score.
TEST_SESSIONS = 10

#: Sessions per class in the training pool the trainer cross-validates over.
POOL_SESSIONS = 20


def ordered_sessions(scene: pathlib.Path) -> list[pathlib.Path]:
    """Every session directory in recording order, which is what the boundaries are relative to."""
    return sorted(scene.glob('session_*'))


def expand_range(sessions: list[pathlib.Path], first: str, last: str) -> list[pathlib.Path]:
    """
    Return the slice of `sessions` from the one matching `first` through the one matching `last`.

    Both boundaries must match exactly one session, and the same string is never resolved twice --
    a suffix that appears in two classes is an error here rather than a silent mislabelling.
    """

    def index_of(pattern: str) -> int:
        hits = [i for i, s in enumerate(sessions) if pattern in s.name]
        if not hits:
            raise ValueError(f'no session matching {pattern!r}')
        if len(hits) > 1:
            raise ValueError(f'{pattern!r} matches {len(hits)}: {[sessions[i].name for i in hits]}')
        return hits[0]

    start, end = index_of(first), index_of(last)
    if end < start:
        raise ValueError(f'range {first}..{last} runs backwards (indices {start} > {end})')
    return sessions[start : end + 1]


def usable(sessions: list[pathlib.Path]) -> list[pathlib.Path]:
    """Drop sessions with no frames -- an aborted recording is not a sample of anything."""
    return [s for s in sessions if any(s.glob('video/frame_*.jpg'))]


def read_session_frames(
    session: pathlib.Path, n_frames: int, img: int, window: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Sample `n_frames` evenly from `window` of a session's JPEGs, crop the mount away, resize.

    The window is a fraction of the recording. Sampling INSIDE it beats sampling across the whole
    session and discarding the ends afterwards, because it keeps the frame budget: the approach and
    release frames look past open jaws at the room, carry no shape, and wear the class label
    anyway, so every one of them kept is a mislabelled sample and every one dropped after sampling
    is a sample not taken.
    """
    files = sorted(session.glob('video/frame_*.jpg'))
    lo = int(len(files) * window[0])
    hi = max(lo + 1, int(round(len(files) * window[1])))
    files = files[lo:hi]
    indices = sample_indices(len(files), n_frames)
    out = np.empty((len(indices), img, img, 3), dtype=np.uint8)
    for i, index in enumerate(indices):
        bgr = cv2.imread(str(files[int(index)]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'could not decode {files[int(index)]}')
        out[i] = cv2.resize(
            crop_finger_rgb(bgr[:, :, ::-1], x_min=CROP_X_MIN), (img, img), interpolation=cv2.INTER_AREA
        )
    return out


def main() -> int:
    """Expand the ranges, sample frames, and write the npz."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', type=pathlib.Path, required=True)
    ap.add_argument('--ranges', type=pathlib.Path, required=True, help='JSON: {"class": ["first", "last"]}')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('texture_ranges.npz'))
    ap.add_argument('--frames-per-session', type=int, default=FRAMES_PER_SESSION)
    ap.add_argument('--test-sessions', type=int, default=TEST_SESSIONS)
    ap.add_argument('--pool-sessions', type=int, default=POOL_SESSIONS)
    ap.add_argument('--img', type=int, default=IMG)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='session substrings to drop entirely, for recordings where the grasp missed',
    )
    ap.add_argument(
        '--window',
        nargs=2,
        type=float,
        default=(0.0, 1.0),
        metavar=('LO', 'HI'),
        help='fraction of each recording to sample from; 0.25 0.85 skips approach and release',
    )
    args = ap.parse_args()

    raw = json.loads(args.ranges.read_text())
    ranges = {k: v for k, v in raw.items() if not k.startswith('_')}
    sessions = ordered_sessions(args.scene)
    print(f'{len(sessions)} sessions in {args.scene.name}\n')
    rng = np.random.default_rng(args.seed)

    images, labels, splits, session_ids = [], [], [], []
    class_names = list(ranges)
    for label_index, label in enumerate(class_names):
        first, last = ranges[label]
        found = expand_range(sessions, first, last)
        with_frames = usable(found)
        good = [s for s in with_frames if not any(p in s.name for p in args.exclude)]
        dropped = len(found) - len(with_frames)
        excluded = len(with_frames) - len(good)

        if len(good) < args.pool_sessions + args.test_sessions:
            print(
                f'error: {label!r} has {len(good)} usable sessions, needs {args.pool_sessions + args.test_sessions}',
                file=sys.stderr,
            )
            return 1

        # Draw the test sessions at random rather than taking the last N: the last sessions of a
        # class are also the latest in time, so a tail split would test on whatever drifted over
        # the run as much as on the shape.
        order = rng.permutation(len(good))
        test_idx = set(order[: args.test_sessions].tolist())
        pool_idx = order[args.test_sessions : args.test_sessions + args.pool_sessions].tolist()

        for i, session in enumerate(good):
            if i in test_idx:
                role = 'test'
            elif i in pool_idx:
                role = 'pool'
            else:
                continue  # surplus session beyond pool + test
            frames = read_session_frames(session, args.frames_per_session, args.img, tuple(args.window))
            images.append(frames)
            labels.extend([label_index] * len(frames))
            splits.extend([role] * len(frames))
            session_ids.extend([session.name] * len(frames))

        print(
            f'{label:>10}: {len(found)} in range, {len(good)} usable'
            + (f' ({dropped} empty dropped)' if dropped else '')
            + (f' ({excluded} excluded)' if excluded else '')
            + f' -> {args.pool_sessions} pool + {args.test_sessions} test'
        )

    x = np.concatenate(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    split = np.asarray(splits)
    session_arr = np.asarray(session_ids)

    np.savez_compressed(
        args.out,
        images=x,
        labels=y,
        split=split,
        classes=np.asarray(class_names),
        scene=session_arr,
        episode=session_arr,
    )

    print(f'\nwrote {args.out}  ({x.shape[0]} frames of {args.img}x{args.img})')
    for name in ('pool', 'test'):
        rows = split == name
        counts = {class_names[c]: int(((y == c) & rows).sum()) for c in range(len(class_names))}
        n_sess = len(set(session_arr[rows]))
        print(f'  {name:>5}: {int(rows.sum()):>5} frames from {n_sess:>3} sessions   {counts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
