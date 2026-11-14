"""
Pull the slip-control channels out of a deployed policy rollout bag into a compact .npz.

These are the CDR/ros2msg bags written during FR3 inference runs of the controlled-slip task.
Decoding the finger camera dominates the runtime and the analysis wants one scalar per frame,
so that reduction happens here, once, and the figures are then cheap to iterate on.

Two clocks are in play and conflating them would invent the result we are trying to measure.
The Pi stamps its camera and audio in epoch nanoseconds at the capture instant and
pi_receiver_node republishes those verbatim, while the /polyumi/diag/* scalars are stamped by
the ROS node on the host. If the Pi and the host disagree, comparing a Pi header stamp against
a host-stamped width would read that skew as a tactile lead. So both clocks are recorded:

  *_t      bag log time  -- host receive clock, the one channel comparisons must use
  *_hdr    header stamp  -- Pi capture clock, kept only to quantify the offset

Log time is the conservative choice for timing: a Pi frame arrives after its capture instant
(network + encode), so using log time biases against finding tactile early. A lead that
survives that bias is real rather than an artifact of clock choice.

Tactile activity is the mean absolute difference between consecutive finger frames over the
crop the policy sees (x >= CROP_X, the fingertip region) after a downsample to 128x128
grayscale -- a motion measure, since what matters for slip is that the finger-object interface
changed, not how bright it is.

Needs the ROS environment, so run it under /usr/bin/python3 with the workspace sourced:

    bash -lc 'source /opt/ros/kilted/setup.bash \
      && python3 analysis/extract_slip_rollout.py --root <dir of trials> --out <dir>'
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CROP_X = 170  # fingertip region; left of this is gripper body
TILE = 128

WIDTH_T = '/polyumi/diag/gripper_width_m'
CMD_T = '/polyumi/target_gripper'
OBSAGE_T = '/polyumi/diag/obs_age_s'
IMAGE_AGE_T = '/polyumi/diag/image_age_s'
AUDIO_T = '/pi/audio/raw'
FINGER_T = '/pi/camera/image/compressed'
TOPICS = (WIDTH_T, CMD_T, OBSAGE_T, IMAGE_AGE_T, AUDIO_T, FINGER_T)


def open_bag(path: pathlib.Path):
    """Open an .mcap rollout bag and return (reader, {topic: type_name})."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader, {t.name: t.type for t in reader.get_all_topics_and_types()}


def header_ns(msg) -> float:
    """Return the source-stamped capture instant, or NaN for messages without a header."""
    h = getattr(msg, 'header', None)
    if h is None:
        return float('nan')
    return float(h.stamp.sec) * 1e9 + float(h.stamp.nanosec)


def tile_of(msg) -> np.ndarray:
    """Decode a CompressedImage to the cropped 128x128 float32 grayscale tile."""
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img[:, CROP_X:], (TILE, TILE)).astype(np.float32) / 255.0


def commanded_width(msg) -> float | None:
    """First commanded gripper position out of a JointTrajectory, if it carries one."""
    if not msg.points or not len(msg.points[0].positions):
        return None
    return float(msg.points[0].positions[0])


def extract(bag: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the bag once, reducing each channel to timestamped arrays."""
    reader, available = open_bag(bag)
    wanted = [t for t in TOPICS if t in available]
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    msg_types = {t: get_message(available[t]) for t in wanted}

    scalars: dict[str, list] = {WIDTH_T: [], CMD_T: [], OBSAGE_T: [], IMAGE_AGE_T: []}
    tactile: list[tuple[float, float, float]] = []
    audio_log: list[float] = []
    audio_hdr: list[float] = []
    audio_pcm: list[np.ndarray] = []
    prev: np.ndarray | None = None

    while reader.has_next():
        topic, raw, log_ns = reader.read_next()
        msg = deserialize_message(raw, msg_types[topic])

        if topic == FINGER_T:
            tile = tile_of(msg)
            if prev is not None:
                tactile.append((float(log_ns), header_ns(msg), float(np.abs(tile - prev).mean())))
            prev = tile
        elif topic == AUDIO_T:
            audio_log.append(float(log_ns))
            audio_hdr.append(header_ns(msg))
            audio_pcm.append(np.frombuffer(bytes(msg.data), dtype=np.int16))
        elif topic == CMD_T:
            w = commanded_width(msg)
            if w is not None:
                scalars[CMD_T].append((float(log_ns), w))
        else:
            scalars[topic].append((float(log_ns), float(msg.data)))

    out: dict[str, np.ndarray] = {}
    for topic, key in ((WIDTH_T, 'width'), (CMD_T, 'cmd'), (OBSAGE_T, 'obsage'), (IMAGE_AGE_T, 'imgage')):
        a = np.array(scalars[topic], dtype=float) if scalars[topic] else np.zeros((0, 2))
        out[f'{key}_t'], out[f'{key}_v'] = a[:, 0], a[:, 1]
    t = np.array(tactile, dtype=float) if tactile else np.zeros((0, 3))
    out['tactile_t'], out['tactile_hdr'], out['tactile_act'] = t[:, 0], t[:, 1], t[:, 2]
    out['audio_t'] = np.array(audio_log, dtype=float)
    out['audio_hdr'] = np.array(audio_hdr, dtype=float)
    out['audio_pcm'] = np.concatenate(audio_pcm) if audio_pcm else np.zeros(0, dtype=np.int16)
    return out


def main() -> None:
    """Extract every trial bag under --root into one .npz apiece."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for bag in sorted(args.root.rglob('*.mcap')):
        arrays = extract(bag)
        np.savez_compressed(args.out / f'{bag.stem}_full.npz', **arrays)
        skew = np.nanmedian(arrays['tactile_t'] - arrays['tactile_hdr']) / 1e6
        print(
            f'  {bag.stem}: {arrays["width_v"].size} width, {arrays["cmd_v"].size} cmd, '
            f'{arrays["tactile_act"].size} tactile, log-minus-header {skew:+.0f} ms'
        )


if __name__ == '__main__':
    main()
