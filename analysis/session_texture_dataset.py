#!/usr/bin/env python3
"""
Build the tactile texture dataset straight from raw Pi sessions, before any ingest step.

``texture_dataset.py`` reads a scene's pzarr; this reads the session directories the Pi writes --
``<scene>/session_*/video/frame_*.jpg`` -- so a scene can be trained on the moment it lands,
with no preprocessing run in between. The frames are the finger camera at its native 1152x648,
which is the resolution the `finger_rgb` crop was measured on, so the same crop applies.

Everything downstream is shared with the pzarr path: the same split, balance and npz layout, so
``train_texture_cnn.py`` reads either without knowing which produced it.

**How the split is drawn depends on how much there is to split.**

* 3 or more sessions in a class -> whole SESSIONS are held out, which is the honest question: a
  shape pressed in a grasp the model never saw.
* fewer -> the frames of each session are split CONTIGUOUSLY in time, first 80% train, then val,
  then test. This is a weak test and is reported as such: frames seconds apart in one continuous
  recording share the object, the grasp and the lighting, so val and test accuracy will read high
  whether or not the model learned the shape. Collect a second session per class and this path
  stops being used automatically.

Usage:

    uv run python analysis/session_texture_dataset.py \
        --scene recordings/scene_2026-09-16_02-37-02_510e \
        --labels analysis/texture_classes.json \
        --out texture_tensors.npz --frames-per-session 20

where the labels name sessions by their directory (a 4-character suffix is enough):

    {"flat": ["4f1d"], "triangle": ["e963"], "N": ["64d9"], "square": ["c229"], "circle": ["51b2"]}

A class may instead name its sessions per role, which is what a second round of the same shapes
is for -- round one trains, round two is the held-out test, and val comes from the tail of the
training recording:

    {"flat": {"train": ["4f1d"], "test": ["9680"]}}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from texture_dataset import CROP_X_MIN, IMG, assign_episode_splits, balance, sample_indices  # noqa: E402

from polyumi_ingest.camera_preproc import crop_finger_rgb  # noqa: E402

#: Frames kept per session, evenly spaced across it. 20 by default: the point of spreading them is
#: that the grasp happens at a different moment in every recording, so no class ends up being
#: represented only by the approach or only by the release.
FRAMES_PER_SESSION = 20

#: Below this many sessions in a class, whole-session holdout is impossible and the split falls
#: back to contiguous frames within each session. Three is the floor: one each for train, val,
#: test.
MIN_SESSIONS_FOR_HOLDOUT = 3


def resolve_sessions(scene: pathlib.Path, patterns: list[str]) -> list[pathlib.Path]:
    """
    Map each pattern to exactly one session directory, matching on any unique substring.

    The Pi's session names carry a timestamp and a 4-character suffix, and the suffix is what a
    person reads off a label, so matching on a substring means the mapping file can say "4f1d"
    rather than the whole name. Ambiguity is an error rather than a first match, since silently
    taking the wrong session would mislabel a whole class.
    """
    resolved = []
    for pattern in patterns:
        matches = sorted(p for p in scene.glob('session_*') if pattern in p.name)
        if not matches:
            raise ValueError(f'no session under {scene} matching {pattern!r}')
        if len(matches) > 1:
            raise ValueError(f'{pattern!r} matches {len(matches)} sessions: {[m.name for m in matches]}')
        resolved.append(matches[0])
    return resolved


def read_session_frames(session: pathlib.Path, n_frames: int, img: int) -> np.ndarray:
    """Sample `n_frames` evenly across a session's JPEGs, crop away the mount, resize to `img`."""
    files = sorted(session.glob('video/frame_*.jpg'))
    if not files:
        raise ValueError(f'{session} holds no video/frame_*.jpg')
    indices = sample_indices(len(files), n_frames)
    out = np.empty((len(indices), img, img, 3), dtype=np.uint8)
    for i, index in enumerate(indices):
        bgr = cv2.imread(str(files[int(index)]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'could not decode {files[int(index)]}')
        cropped = crop_finger_rgb(bgr[:, :, ::-1], x_min=CROP_X_MIN)
        out[i] = cv2.resize(cropped, (img, img), interpolation=cv2.INTER_AREA)
    return out


def contiguous_split(n: int, val_frac: float, test_frac: float) -> list[str]:
    """
    Label `n` time-ordered frames train/val/test in contiguous blocks, val and test at the end.

    Contiguous rather than random because the frames come from one continuous recording: a random
    split puts frame k in train and frame k+1 in val, which are the same picture to within a
    tenth of a second. Taking the tail as val and test at least separates them in time, and in
    what the hand was doing. It does not make them independent.
    """
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    n_train = max(1, n - n_val - n_test)
    return ['train'] * n_train + ['val'] * n_val + ['test'] * n_test


def main() -> int:
    """Build the npz from raw sessions."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', type=pathlib.Path, required=True, help='scene directory holding session_* dirs')
    ap.add_argument('--labels', type=pathlib.Path, required=True, help='JSON: {"class": ["session substring", ...]}')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('texture_tensors.npz'))
    ap.add_argument('--frames-per-session', type=int, default=FRAMES_PER_SESSION)
    ap.add_argument('--img', type=int, default=IMG)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--test-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-balance', action='store_true', help='keep every frame instead of equalising classes')
    args = ap.parse_args()

    raw = json.loads(args.labels.read_text())
    # dict(v), not list(v), for the per-role form: list() on a dict yields its KEYS, which turns
    # {"train": [...], "test": [...]} into ["train", "test"] and sends the resolver looking for a
    # session called "train".
    mapping = {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in raw.items() if not k.startswith('_')}
    class_names = sorted(mapping)
    rng = np.random.default_rng(args.seed)

    images, labels, splits, session_ids, episode_ids = [], [], [], [], []
    weak = []

    def add(label_index: int, session: pathlib.Path, frames: np.ndarray, assignment: list[str]) -> None:
        """Append one session's frames under the split labels given for them."""
        images.append(frames)
        labels.extend([label_index] * len(frames))
        splits.extend(assignment)
        session_ids.extend([session.name] * len(frames))
        episode_ids.extend([session.name] * len(frames))

    for label_index, label in enumerate(class_names):
        spec = mapping[label]

        if isinstance(spec, dict):
            # Explicit form: the operator says which recording is train and which is test, which
            # is what a second round of the same shapes is for. Val comes out of the tail of the
            # training sessions unless its own sessions are named, since with one recording per
            # role there is nothing else to take it from.
            by_role = {role: resolve_sessions(args.scene, spec.get(role, [])) for role in ('train', 'val', 'test')}
            if not by_role['train'] or not by_role['test']:
                print(f'error: class {label!r} needs both "train" and "test" sessions', file=sys.stderr)
                return 1
            n_sessions = sum(len(v) for v in by_role.values())
            total = 0
            for role, sessions in by_role.items():
                for session in sessions:
                    frames = read_session_frames(session, args.frames_per_session, args.img)
                    total += len(frames)
                    if role == 'train' and not by_role['val']:
                        add(label_index, session, frames, contiguous_split(len(frames), args.val_frac, 0.0))
                    else:
                        add(label_index, session, frames, [role] * len(frames))
        else:
            sessions = resolve_sessions(args.scene, spec)
            sampled = {s: read_session_frames(s, args.frames_per_session, args.img) for s in sessions}
            n_sessions, total = len(sessions), sum(len(v) for v in sampled.values())
            if len(sessions) >= MIN_SESSIONS_FOR_HOLDOUT:
                episodes = [(args.scene.name, s.name, len(sampled[s])) for s in sessions]
                assignment = assign_episode_splits(episodes, args.val_frac, args.test_frac, rng)
                for session in sessions:
                    add(
                        label_index,
                        session,
                        sampled[session],
                        [assignment[(args.scene.name, session.name)]] * len(sampled[session]),
                    )
            else:
                weak.append(label)
                for session in sessions:
                    frames = sampled[session]
                    add(label_index, session, frames, contiguous_split(len(frames), args.val_frac, args.test_frac))

        print(f'{label:>12}: {n_sessions} session(s), {total} frames sampled')

    x = np.concatenate(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    split = np.asarray(splits)
    session_arr = np.asarray(session_ids)
    episode_arr = np.asarray(episode_ids)

    if not args.no_balance:
        keep = balance(y, split, rng)
        x, y, split, session_arr, episode_arr = (a[keep] for a in (x, y, split, session_arr, episode_arr))

    np.savez_compressed(
        args.out,
        images=x,
        labels=y,
        split=split,
        classes=np.asarray(class_names),
        scene=session_arr,
        episode=episode_arr,
    )

    print(f'\nwrote {args.out}  ({x.shape[0]} frames of {args.img}x{args.img}, {x.nbytes / 1e6:.0f} MB uncompressed)')
    for name in ('train', 'val', 'test'):
        rows = split == name
        counts = {class_names[c]: int(((y == c) & rows).sum()) for c in range(len(class_names))}
        print(f'  {name:>5}: {int(rows.sum()):>5} frames   {counts}')
    if weak:
        print(
            f'\nWARNING: {", ".join(weak)} had fewer than {MIN_SESSIONS_FOR_HOLDOUT} sessions, so their\n'
            '  val/test frames come from the SAME recording as their training frames, just later in\n'
            '  it. Treat the resulting accuracy as an upper bound, not as evidence the model reads\n'
            '  the shape: a second session per class is what turns this into a real test.'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
