#!/usr/bin/env python3
"""
Render the contact-mic audio of each shake trial as a full-length log-mel image.

One PNG per trial covering the WHOLE recording at native time resolution, so the shake can be
located by eye and a crop window chosen from it. Nothing is trimmed and the time axis is never
resampled: stretching a trial to a fixed width is what makes two recordings look alike when they
are not, and it would destroy the mapping from pixels to seconds that choosing a crop depends on.

Each image carries a second-by-second ruler along the bottom, so a crop is read straight off it.

Frequency runs low at the bottom, as in the Vista previews, and the colour ramp is the same
magma-like LUT they use. Contrast is stretched per trial between the 1st and 99.5th percentile:
a shared scale would hide the quiet trials, and the point here is to see where the motion is, not
to compare loudness across trials.

A per-class contact sheet is also written, stacking every trial of a class on one shared time
axis -- that is what shows whether the shake lands in the same window across a class.

Usage (ROS sourced, for rosbag2_py):
    /usr/bin/python3 analysis/mcap_logmel_images.py \
        --root "/media/.../PolyUmi datasets/water_material_rv4" --out water_logmel
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from water_dataset import AUDIO_TOPIC, HOP, N_FFT, N_MELS, log_mel, mel_filterbank  # noqa: E402

#: Vertical blow-up of the 64 mel bins, so a spectrogram is legible rather than a thin strip.
MEL_SCALE = 4

#: Seconds between ruler ticks along the bottom of each image.
TICK_S = 1.0


def magma_rgb(x: np.ndarray) -> np.ndarray:
    """Map [0, 1] to RGB uint8 with the same magma-like LUT the Vista mel previews use."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.7 * x - 0.2, 0, 1)
    g = np.clip(1.5 * x - 0.5, 0, 1)
    b = np.clip(2.0 * x, 0, 1) * (1.0 - 0.6 * x)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def read_audio(bag: pathlib.Path) -> tuple[np.ndarray, int]:
    """
    Concatenate the whole bag's contact-mic PCM. Returns (samples in [-1, 1], sample rate).

    Only audio messages are deserialised -- the bags also carry two camera streams, and decoding
    those would cost minutes per class for frames this tool never looks at.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if AUDIO_TOPIC not in types:
        return np.zeros(0), 16000

    chunks, sr = [], 16000
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != AUDIO_TOPIC:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        sr = int(msg.sample_rate)
        samples = np.frombuffer(bytes(msg.data), dtype='<i2').astype(np.float64) / 32768.0
        if int(msg.number_of_channels) > 1:  # interleaved -> keep one channel
            samples = samples[:: int(msg.number_of_channels)]
        chunks.append(samples)
    return (np.concatenate(chunks) if chunks else np.zeros(0)), sr


def render(lm: np.ndarray, scale: int = MEL_SCALE) -> np.ndarray:
    """Colour a log-mel array into an RGB image, low frequency at the bottom, bins blown up."""
    lo, hi = np.percentile(lm, 1.0), np.percentile(lm, 99.5)
    norm = (lm - lo) / max(hi - lo, 1e-6)
    rgb = magma_rgb(norm[::-1])  # flip so low frequency sits at the bottom
    return np.repeat(rgb, scale, axis=0)


def add_ruler(img: np.ndarray, sr: int, hop: int, tick_s: float = TICK_S) -> np.ndarray:
    """
    Add a seconds ruler under the spectrogram so a crop window can be read straight off it.

    Every second is labelled, not just multiples of five: the point of these images is to choose a
    crop, and counting unlabelled lines across a 20 s recording is the error the ruler exists to
    prevent. Minor gridlines are dimmed so they mark time without hiding the signal.
    """
    import cv2

    frames_per_s = sr / hop
    strip = np.full((26, img.shape[1], 3), 20, dtype=np.uint8)
    out = np.concatenate([img, strip], axis=0)
    seconds = img.shape[1] / frames_per_s
    for t in np.arange(0, seconds, tick_s):
        px = int(round(t * frames_per_s))
        if px >= img.shape[1]:
            break
        major = int(round(t)) % 5 == 0
        colour = (255, 255, 255) if major else (110, 110, 110)
        cv2.line(out, (px, 0), (px, img.shape[0]), colour, 1, cv2.LINE_AA)
        cv2.putText(
            out,
            f'{t:.0f}s' if major else f'{t:.0f}',
            (px + 2, img.shape[0] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    """Render every trial under --root."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=pathlib.Path, required=True, help='dataset root holding one directory per class')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('water_logmel'))
    ap.add_argument('--f-min', type=float, default=20.0, help='low edge of the mel filterbank, Hz')
    ap.add_argument('--scale', type=int, default=MEL_SCALE)
    ap.add_argument('--limit', type=int, default=0, help='render only this many trials per class (0 = all)')
    args = ap.parse_args()

    import cv2

    classes = sorted(p for p in args.root.iterdir() if p.is_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{"class":>18} {"trial":>10} {"seconds":>8} {"frames":>8} {"width px":>9}')

    for class_dir in classes:
        trials = sorted(
            (t for t in class_dir.iterdir() if t.is_dir()),
            key=lambda p: int(p.name.rsplit('_', 1)[1]) if p.name.rsplit('_', 1)[1].isdigit() else 0,
        )
        if args.limit:
            trials = trials[: args.limit]
        (args.out / class_dir.name).mkdir(parents=True, exist_ok=True)
        sheet_rows, sheet_labels = [], []

        for trial in trials:
            bags = sorted(trial.glob('*.mcap'))
            if not bags:
                continue
            samples, sr = read_audio(bags[0])
            if samples.size < N_FFT:
                print(f'{class_dir.name:>18} {trial.name:>10}   no audio')
                continue
            fb = mel_filterbank(N_FFT, N_MELS, sr, args.f_min)
            lm = log_mel(samples, sr, fb)
            img = add_ruler(render(lm, args.scale), sr, HOP)
            cv2.imwrite(str(args.out / class_dir.name / f'{trial.name}.png'), img[:, :, ::-1])
            print(f'{class_dir.name:>18} {trial.name:>10} {samples.size / sr:>8.1f} {lm.shape[1]:>8} {img.shape[1]:>9}')
            sheet_rows.append(render(lm, 1))  # unscaled bins keep the contact sheet compact
            sheet_labels.append(trial.name)

        # One contact sheet per class: every trial on a shared time axis, so a window that holds
        # the shake in all of them can be picked at a glance.
        if sheet_rows:
            width = max(r.shape[1] for r in sheet_rows)
            padded = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in sheet_rows]
            gap = np.zeros((2, width, 3), dtype=np.uint8)
            stack = np.concatenate([np.concatenate([p, gap], axis=0) for p in padded], axis=0)
            stack = add_ruler(stack, sr, HOP)
            cv2.imwrite(str(args.out / f'{class_dir.name}_all_trials.png'), stack[:, :, ::-1])
            print(f'{class_dir.name:>18}  contact sheet: {len(sheet_rows)} trials -> {stack.shape[1]}x{stack.shape[0]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
