"""
Pull the slip-control channels out of the human demonstration bags into compact .npz files.

These bags are the recorded demonstrations of the controlled-slip task (the object starts at a
set angle and is brought to 90 degrees by modulating grip width so it slides a little at a
time). They are stored as foxglove JSON schemas rather than CDR, so no ROS environment is
needed -- the mcap reader plus json is enough, and this runs under the workspace venv.

Unlike the policy rollout bags there is no commanded-width channel, because the demonstrator
is the controller. Every width release here is therefore an intentional control action, which
makes the event set cleaner than the rollouts' commanded/uncommanded split.

Tactile activity is the mean absolute difference between consecutive finger frames over the
crop the policy sees (x >= CROP_X, the fingertip region) after a downsample to 128x128
grayscale -- a motion measure, since what matters for slip is that the finger-object interface
changed, not how bright it is.

    uv run python analysis/extract_slip_demo.py --root <dir of scenes> --out <dir>
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib

import cv2
import numpy as np
from mcap.reader import make_reader

CROP_X = 170  # fingertip region; left of this is gripper body
TILE = 128
TOPICS = ('/gripper/width', '/finger/image', '/finger/piezo')


def stamp_ns(msg: dict) -> int:
    """Capture-instant timestamp, which the Pi stamps at source rather than at log time."""
    t = msg['timestamp']
    return int(t['sec']) * 1_000_000_000 + int(t['nsec'])


def tile_of(payload: str) -> np.ndarray:
    """Decode a base64 JPEG to the cropped 128x128 float32 grayscale tile."""
    buf = np.frombuffer(base64.b64decode(payload), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img[:, CROP_X:], (TILE, TILE)).astype(np.float32) / 255.0


def extract(bag: pathlib.Path) -> dict[str, np.ndarray] | None:
    """Reduce one episode to per-channel arrays, or None if it lacks the channels we need."""
    width: list[tuple[int, float]] = []
    tactile: list[tuple[int, float]] = []
    audio_t: list[int] = []
    audio_pcm: list[np.ndarray] = []
    prev: np.ndarray | None = None
    raw_shape: tuple[int, int] | None = None

    with bag.open('rb') as fh:
        for _schema, channel, message in make_reader(fh).iter_messages(topics=list(TOPICS)):
            msg = json.loads(message.data)
            if channel.topic == '/gripper/width':
                width.append((stamp_ns(msg), float(msg['width_m'])))
            elif channel.topic == '/finger/image':
                if raw_shape is None:
                    probe = cv2.imdecode(
                        np.frombuffer(base64.b64decode(msg['data']), np.uint8),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    raw_shape = probe.shape
                tile = tile_of(msg['data'])
                if prev is not None:
                    tactile.append((stamp_ns(msg), float(np.abs(tile - prev).mean())))
                prev = tile
            elif channel.topic == '/finger/piezo':
                audio_t.append(stamp_ns(msg))
                audio_pcm.append(np.frombuffer(base64.b64decode(msg['data']), dtype=np.int16))

    if len(width) < 50 or len(tactile) < 20 or not audio_pcm:
        return None

    w = np.array(width, dtype=float)
    t = np.array(tactile, dtype=float)
    return {
        'width_t': w[:, 0],
        'width_v': w[:, 1],
        'tactile_t': t[:, 0],
        'tactile_act': t[:, 1],
        'audio_t': np.array(audio_t, dtype=float),
        'audio_pcm': np.concatenate(audio_pcm),  # mono piezo contact mic
        'finger_shape': np.array(raw_shape),
    }


def main() -> None:
    """Extract every episode under --root into one .npz apiece."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bags = sorted(args.root.rglob('*.mcap'))
    kept = 0
    for bag in bags:
        arrays = extract(bag)
        if arrays is None:
            print(f'  skip {bag.parent.name}/{bag.name} (missing channels)')
            continue
        name = f'{bag.parent.name}_{bag.stem}.npz'
        np.savez_compressed(args.out / name, **arrays)
        kept += 1
        print(
            f'  {name}: {arrays["width_v"].size} width, {arrays["tactile_act"].size} tactile, '
            f'finger {tuple(arrays["finger_shape"])}'
        )
    print(f'\n{kept} of {len(bags)} episodes extracted to {args.out}')


if __name__ == '__main__':
    main()
