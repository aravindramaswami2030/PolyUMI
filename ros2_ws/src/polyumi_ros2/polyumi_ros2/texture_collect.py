#!/usr/bin/env python3
"""
Collect a labelled finger-camera texture dataset, one keypress at a time.

The operator drives the loop; the arm and the gripper do the parts that have to be identical
across 250 samples. Per sample:

    1. the arm moves to a fresh random pose (position AND orientation)
    2. ENTER -> the gripper opens fully, so the object can be placed or swapped
    3. ENTER -> the gripper closes to the ONE width this whole dataset is gripped at
    4. ENTER -> the finger camera frame is saved as a PNG under its class

The object to fetch is chosen for you and printed before step 2 -- a random object from the class
whose turn it is -- so the operator never decides what goes in the gripper. What is collected is
therefore a plan, not a running choice, which is what keeps the classes balanced and their
collection order interleaved.

**Layout on disk** is torchvision ``ImageFolder``'s: ``<root>/<class>/<class>_<object>_<n>.png``,
plus ``<root>/manifest.jsonl``, one JSON object per sample carrying the object, the TCP pose, the
grip width and the crop -- everything the image alone cannot say. A simple vision CNN trains off
the directory tree; the manifest is there to audit a result after the fact (e.g. whether an
object, rather than its class, is what a model latched onto).

**Interleaving** matters more than it looks, and is not optional. Collected one class at a time,
any nuisance that drifts over a session -- the finger camera's level and colour balance settle
differently each time the rig is disturbed -- lines up exactly with the label, and a classifier
reads that instead of the texture. On the water corpus that scored 93% on a modality carrying no
information. So the plan runs in blocks: `block` samples of each class in turn, then round two,
and so on. Randomising the pose masks such an offset; only interleaving removes it.

**One grip width for every object**, calibrated once and then commanded outright, for the same
reason: a width measured per object makes the grip a perfect predictor of the class. It is
written to ``<root>/grip_width_m`` and reused by every later run, as is the position the poses
are jittered about (``<root>/home_xyz``).

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: bringup + inference, arm execution on.
    ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true

    # 2. DRY RUN (default): nothing moves, poses go to /polyumi/target_poses_preview.
    ros2 run polyumi_ros2 texture_collect --ros-args \
        -p config:=/path/to/texture_objects.json -p root:=/data/texture_dataset

    # 3. When the preview looks right, collect for real:
    ros2 run polyumi_ros2 texture_collect --ros-args -p execute:=true \
        -p config:=/path/to/texture_objects.json -p root:=/data/texture_dataset

Re-running resumes: each class's next sample number is counted off disk, so an interrupted
session continues where it stopped rather than overwriting.

**Do not run this while policy_client_node is running.** Both publish to
/polyumi/target_poses_traj and the controller splices whichever chunk arrives last.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import threading
import time

from builtin_interfaces.msg import Duration
import cv2
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from polyumi_ros2.camera_preproc import crop_finger_rgb
from polyumi_ros2.gripper_map import aperture_from_joint_state
from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher
from polyumi_ros2.water_shake import level_rotation
from tf2_ros import Buffer, TransformListener

FINGER_TOPIC = '/pi/camera/image/compressed'
GRIPPER_TOPIC = '/polyumi/target_gripper'
GRIPPER_STATE_TOPIC = '/fr3_gripper/joint_states'
GRIPPER_JOINT_NAME = 'fr3_gripper_width'
PREVIEW_TOPIC = '/polyumi/target_poses_preview'

#: Caliper measurement of the PolyUMI fingers' full stroke (see gripper_map.py). "Fully open" is
#: this, not the stock attachments' 0.105 m, which these fingers cannot reach.
MAX_APERTURE_M = 0.0812

#: Same speed ceiling the shake respects, from nuc/config/polyumi_controllers.yaml. A move asked
#: for faster than this is stretched by the controller rather than truncated, so the node warns
#: instead of silently taking longer than the operator expects.
MAX_SPEED_MPS = 1.0


def load_classes(path: pathlib.Path) -> dict[str, list[str]]:
    """
    Read the ``{"class": ["object", ...]}`` plan, rejecting anything a directory name cannot hold.

    Objects are named per class rather than globally so the same physical object may legitimately
    appear in two classes (a mug is "ceramic" whichever mug it is), and so the printed instruction
    can name one specific thing to pick up.
    """
    raw = json.loads(pathlib.Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f'{path}: expected a non-empty object of "class": [objects], got {type(raw).__name__}')
    classes: dict[str, list[str]] = {}
    for label, objects in raw.items():
        # Keys starting with _ are notes to the operator, not classes. A comment is a list of
        # sentences, which would otherwise pass the list check and then fail the filename check
        # on its spaces -- refusing to start over the documentation shipped alongside the plan.
        if label.startswith('_'):
            continue
        if not isinstance(objects, list) or not objects:
            raise ValueError(f'{path}: class {label!r} needs a non-empty list of object names')
        for name in [label, *objects]:
            if not isinstance(name, str) or not name or not all(c.isalnum() or c in '_-' for c in name):
                raise ValueError(f'{path}: {name!r} must be non-empty and [A-Za-z0-9_-] only -- it becomes a filename')
        classes[label] = list(objects)
    return classes


def build_plan(
    classes: dict[str, list[str]], n_per_class: int, block: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """
    Order the whole session as (class, object) pairs: `block` of each class in turn, then repeat.

    The object is drawn uniformly per sample rather than cycled, so no object lands at a fixed
    position in the block order -- otherwise "third sample of every block" would be one object and
    a model could learn the schedule. Every class ends up with exactly `n_per_class` samples; the
    last round takes the remainder when `n_per_class` is not a multiple of `block`.
    """
    if n_per_class < 1 or block < 1:
        raise ValueError(f'n_per_class and block must both be >= 1, got {n_per_class} and {block}')
    labels = list(classes)
    remaining = {label: n_per_class for label in labels}
    plan: list[tuple[str, str]] = []
    while any(remaining.values()):
        for label in labels:
            take = min(block, remaining[label])
            for _ in range(take):
                objects = classes[label]
                plan.append((label, objects[int(rng.integers(len(objects)))]))
            remaining[label] -= take
    return plan


def randomized_pose(
    reference_xyz: np.ndarray,
    level_quat: np.ndarray,
    rng: np.random.Generator,
    jitter_xyz_m: np.ndarray,
    yaw_deg: float,
    tilt_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw one sample pose: the reference position jittered per axis, the levelled tool turned.

    Orientation is randomised in two pieces with different meanings. Yaw (about world Z) swings
    the tool around the approach direction and is the cheap, safe variation -- it changes what the
    finger camera sees behind the object without changing how the object sits in the jaws. Tilt is
    a rotation of at most `tilt_deg` about a random horizontal axis, kept small because past a few
    degrees the object's own weight starts to shift it in the grip, which is a change in the
    sample rather than in the viewpoint.

    Both are bounded, not unbounded random rotations: a uniformly random orientation would put the
    arm in poses it cannot reach and point the tool into the table.
    """
    offset = rng.uniform(-np.abs(jitter_xyz_m), np.abs(jitter_xyz_m))
    yaw = math.radians(yaw_deg) * rng.uniform(-1.0, 1.0)
    tilt_axis_angle = rng.uniform(0.0, 2.0 * math.pi)
    tilt = math.radians(tilt_deg) * rng.uniform(-1.0, 1.0)
    axis = np.array([math.cos(tilt_axis_angle), math.sin(tilt_axis_angle), 0.0])
    rotation = Rotation.from_rotvec([0.0, 0.0, yaw]) * Rotation.from_rotvec(axis * tilt)
    return np.asarray(reference_xyz, dtype=float) + offset, (rotation * Rotation.from_quat(level_quat)).as_quat()


def sample_path(root: pathlib.Path, label: str, obj: str, index: int) -> pathlib.Path:
    """Where sample `index` of `label` lands: one directory per class, as ImageFolder wants."""
    return pathlib.Path(root) / label / f'{label}_{obj}_{index:04d}.png'


def next_index(root: pathlib.Path, label: str) -> int:
    """
    Sample number to write next for `label`, counted off disk so an interrupted run resumes.

    Read from the filenames rather than from a count of files: a gap left by a deleted bad sample
    must not make the next write land on an existing one.
    """
    existing = sorted((pathlib.Path(root) / label).glob(f'{label}_*.png'))
    numbers = [int(p.stem.rsplit('_', 1)[1]) for p in existing if p.stem.rsplit('_', 1)[1].isdigit()]
    return max(numbers, default=0) + 1


def manifest_row(
    label: str,
    obj: str,
    index: int,
    path: pathlib.Path,
    position: np.ndarray,
    quat: np.ndarray,
    grip_width_m: float,
    crop: dict,
    shape: tuple[int, ...],
) -> dict:
    """Build the JSONL record for one sample: everything the PNG itself cannot carry."""
    return {
        'class': label,
        'object': obj,
        'index': index,
        'file': str(pathlib.Path(path).relative_to(pathlib.Path(path).parents[1])),
        'stamp_s': time.time(),
        'tcp_position_xyz': [float(v) for v in position],
        'tcp_orientation_xyzw': [float(v) for v in quat],
        'grip_width_m': float(grip_width_m),
        'finger_crop': crop,
        'image_hwc': [int(v) for v in shape],
    }


def _pose(position: np.ndarray, quat: np.ndarray) -> Pose:
    """Build a geometry_msgs Pose from a 3-vector and an xyzw quaternion."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
    return p


class TextureCollectNode(Node):
    """Drive the arm and gripper through a planned texture collection, saving one image per sample."""

    def __init__(self, **kwargs):
        """Declare parameters, open the publishers, and subscribe to the finger camera."""
        super().__init__('texture_collect', **kwargs)
        self.declare_parameter('config', '')
        self.declare_parameter('root', '/data/texture_dataset')
        self.declare_parameter('samples_per_class', 50)
        # Samples of one class before swapping to the next. 10 gives five rounds over a 50-sample
        # class; smaller interleaves harder at the cost of more object swaps.
        self.declare_parameter('block', 10)
        self.declare_parameter('seed', -1)
        # Bounds of the per-sample position jitter, and of the two orientation pieces. See
        # randomized_pose for why tilt stays small while yaw does not have to.
        self.declare_parameter('jitter_xyz_m', [0.04, 0.04, 0.03])
        self.declare_parameter('yaw_deg', 25.0)
        self.declare_parameter('tilt_deg', 8.0)
        self.declare_parameter('home_xyz', [0.0, 0.0, 0.0])
        # One width for the whole dataset. 0.0 means "calibrate on the first grip and reuse".
        self.declare_parameter('grip_width_m', 0.0)
        self.declare_parameter('grip_squeeze_m', 0.002)
        self.declare_parameter('open_width_m', MAX_APERTURE_M)
        self.declare_parameter('grip_time_s', 1.0)
        self.declare_parameter('move_time_s', 3.0)
        self.declare_parameter('waypoint_dt', 0.05)
        self.declare_parameter('settle_s', 0.5)
        # The finger_rgb crop contract (ingest/config/finger_camera.yaml): the mount occludes the
        # left of the view, and 170 is the first column that is entirely image. Saving the crop
        # rather than the raw frame keeps a classifier from scoring on the mount.
        self.declare_parameter('finger_crop.x_min', 170)
        self.declare_parameter('finger_crop.x_max', -1)
        self.declare_parameter('finger_crop.y_min', 0)
        self.declare_parameter('finger_crop.y_max', -1)
        self.declare_parameter('finger_topic', FINGER_TOPIC)
        # Off by default, exactly as water_shake: a dry run publishes the poses to the preview
        # topic and moves nothing, so a bad reference position is seen before the arm takes it.
        self.declare_parameter('execute', False)
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')

        self._p = {
            name: self.get_parameter(name).value
            for name in (
                'config',
                'root',
                'samples_per_class',
                'block',
                'seed',
                'jitter_xyz_m',
                'yaw_deg',
                'tilt_deg',
                'home_xyz',
                'grip_width_m',
                'grip_squeeze_m',
                'open_width_m',
                'grip_time_s',
                'move_time_s',
                'waypoint_dt',
                'settle_s',
                'finger_topic',
                'execute',
                'base_frame',
                'eef_frame',
            )
        }
        self._crop = {}
        for bound in ('x_min', 'x_max', 'y_min', 'y_max'):
            value = self.get_parameter(f'finger_crop.{bound}').get_parameter_value().integer_value
            self._crop[bound] = None if value < 0 else value

        self._root = pathlib.Path(self._p['root'])
        self._grip_lock = threading.Lock()
        self._grip_width: float | None = None
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None

        self.create_subscription(JointState, GRIPPER_STATE_TOPIC, self._on_gripper_state, 10)
        self.create_subscription(CompressedImage, self._p['finger_topic'], self._on_finger, 10)
        self._grip_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._preview = self.create_publisher(PoseArray, PREVIEW_TOPIC, 10)
        self._pub = (
            TargetChunkPublisher(self, frame_id=self._p['base_frame'], joint_name=self._p['eef_frame'])
            if self._p['execute']
            else None
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _on_gripper_state(self, msg: JointState) -> None:
        """Cache the jaw aperture reported by whichever gripper driver is running."""
        width = aperture_from_joint_state(msg)
        if width is not None:
            with self._grip_lock:
                self._grip_width = float(width)

    def _on_finger(self, msg: CompressedImage) -> None:
        """Decode and crop the newest finger frame, keeping only the latest."""
        decoded = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return
        # imdecode gives BGR; crop_finger_rgb is indifferent to channel order, and the RGB array
        # is what gets written back out through cv2.imwrite below.
        frame = crop_finger_rgb(decoded[:, :, ::-1], **self._crop)
        with self._frame_lock:
            self._frame = frame

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _command_width(self, target: float) -> float:
        """Publish one absolute jaw width and report it in a form a script can capture."""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        point.time_from_start = Duration(sec=int(self._p['grip_time_s']), nanosec=0)
        msg.points.append(point)
        self._grip_pub.publish(msg)
        self.get_logger().info(f'GRIP_TARGET_M={target:.6f}')
        return float(target)

    def open_gripper(self) -> None:
        """Open the jaws to their full stroke so the object can be placed or swapped."""
        if not self._p['execute']:
            self.get_logger().info(f'dry run: would open to {self._p["open_width_m"] * 1000:.1f}mm')
            return
        self._command_width(float(self._p['open_width_m']))
        time.sleep(float(self._p['grip_time_s']))

    def close_gripper(self) -> float | None:
        """
        Close to the dataset's shared width, calibrating it once if none is on record.

        Calibration reads where the jaws come to rest on the object and takes `grip_squeeze_m` off
        it. That happens at most once per dataset: the width is logged, written to disk by the
        wrapper script, and passed back in as `grip_width_m` for every sample after, so no class
        gets its own grip.
        """
        if not self._p['execute']:
            self.get_logger().info('dry run: would close the gripper')
            return None
        fixed = float(self._p['grip_width_m'])
        if fixed > 0.0:
            width = self._command_width(fixed)
            time.sleep(float(self._p['grip_time_s']))
            return width

        deadline = time.monotonic() + 10.0
        resting = None
        while time.monotonic() < deadline:
            with self._grip_lock:
                resting = self._grip_width
            if resting is not None:
                break
            time.sleep(0.1)
        if resting is None:
            self.get_logger().warn(f'no {GRIPPER_STATE_TOPIC} -- not gripping. Is the driver up?')
            return None
        target = max(0.0, resting - float(self._p['grip_squeeze_m']))
        self.get_logger().info(
            f'gripper: calibrating -- resting at {resting * 1000:.1f}mm, commanding {target * 1000:.1f}mm. '
            f'Pass grip_width_m:={target:.6f} for every sample after this one.'
        )
        width = self._command_width(target)
        time.sleep(float(self._p['grip_time_s']))
        self._p['grip_width_m'] = width
        return width

    # ------------------------------------------------------------------
    # Arm
    # ------------------------------------------------------------------

    def lookup_tcp(self, timeout_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (position, xyzw quaternion) of the TCP in the base frame.

        Polls `can_transform` rather than leaning on the lookup's own timeout, because until the
        TF subscription is matched across the Humble NUC / Kilted host boundary the buffer holds
        no frames at all and the lookup reports "frame does not exist" -- which reads like a
        crashed bringup and is really an unfinished handshake.
        """
        base, eef = self._p['base_frame'], self._p['eef_frame']
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._tf_buffer.can_transform(base, eef, rclpy.time.Time()):
                tf = self._tf_buffer.lookup_transform(base, eef, rclpy.time.Time())
                t, r = tf.transform.translation, tf.transform.rotation
                return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])
            time.sleep(0.1)
        known = self._tf_buffer.all_frames_as_string() or '(none)'
        raise RuntimeError(f'TF {base} -> {eef} did not appear within {timeout_s:.0f}s.\nFrames seen:\n{known}')

    def reference_position(self, measured: np.ndarray) -> np.ndarray:
        """
        Return the point every sample pose is jittered about: `home_xyz`, else where the arm is.

        Jittering about the CURRENT pose instead would make each sample start from the last one's
        offset, so the start would random-walk across 250 samples and drift would end up aligned
        with collection order -- the very thing the interleaving is there to prevent.
        """
        home = [float(v) for v in self._p['home_xyz']]
        reference = np.array(home) if len(home) == 3 and any(home) else np.asarray(measured, dtype=float)
        self.get_logger().info(f'HOME_XYZ={reference[0]:.6f},{reference[1]:.6f},{reference[2]:.6f}')
        return reference

    def move_to(self, position: np.ndarray, quat: np.ndarray) -> None:
        """Command one straight move from the current pose to (position, quat), or preview it."""
        current_p, current_q = self.lookup_tcp()
        dt = float(self._p['waypoint_dt'])
        n = max(2, int(round(float(self._p['move_time_s']) / dt)))
        travel = float(np.linalg.norm(position - current_p))
        if travel / max(float(self._p['move_time_s']), 1e-6) > MAX_SPEED_MPS:
            self.get_logger().warn(
                f'move of {travel * 100:.1f}cm in {self._p["move_time_s"]:.1f}s exceeds '
                f'{MAX_SPEED_MPS} m/s -- the controller will stretch it. Lengthen move_time_s.'
            )
        ramp = np.linspace(0.0, 1.0, n)[:, None]
        path = current_p + ramp * (np.asarray(position, dtype=float) - current_p)
        from scipy.spatial.transform import Slerp

        quats = Slerp([0.0, 1.0], Rotation.from_quat(np.vstack([current_q, quat])))(np.linspace(0.0, 1.0, n)).as_quat()
        poses = [_pose(p, q) for p, q in zip(path, quats)]
        poses += [_pose(position, quat)] * max(0, int(round(float(self._p['settle_s']) / dt)))

        preview = PoseArray()
        preview.header.frame_id = self._p['base_frame']
        preview.header.stamp = self.get_clock().now().to_msg()
        preview.poses = poses
        self._preview.publish(preview)
        if self._pub is None:
            self.get_logger().info(f'DRY RUN -- {len(poses)} waypoints on {PREVIEW_TOPIC}, nothing commanded.')
            return
        self._pub.publish(poses, dt=dt)
        time.sleep(len(poses) * dt)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, label: str, obj: str, index: int, position: np.ndarray, quat: np.ndarray) -> pathlib.Path | None:
        """
        Save the newest finger frame as this sample's PNG and append its manifest row.

        Returns None when no frame has arrived, so the caller can retry the same sample rather
        than advance the counter and leave a hole in the class.
        """
        with self._frame_lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            self.get_logger().error(
                f'no frame on {self._p["finger_topic"]} -- is the Pi streaming (`polyumi-pi stream`)?'
            )
            return None
        path = sample_path(self._root, label, obj, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame[:, :, ::-1])  # back to BGR for imwrite
        row = manifest_row(
            label, obj, index, path, position, quat, float(self._p['grip_width_m']), self._crop, frame.shape
        )
        with (self._root / 'manifest.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        return path


def _prompt(message: str) -> None:
    """Block until ENTER. Plain input() rather than raw keypresses, so this works over ssh/tmux."""
    input(message)


def run_session(node: TextureCollectNode) -> int:
    """Walk the whole plan, prompting per sample. Returns the number of samples saved."""
    params = node._p  # noqa: SLF001 - the node is this function's own collaborator
    classes = load_classes(pathlib.Path(params['config']))
    seed = int(params['seed'])
    rng = np.random.default_rng(None if seed < 0 else seed)
    plan = build_plan(classes, int(params['samples_per_class']), int(params['block']), rng)
    jitter = np.abs(np.array([float(v) for v in params['jitter_xyz_m']], dtype=float))

    measured_p, measured_q = node.lookup_tcp()
    reference = node.reference_position(measured_p)
    level_quat = level_rotation(Rotation.from_quat(measured_q).as_matrix()[:, 2]).as_quat()

    done = {label: next_index(node._root, label) - 1 for label in classes}  # noqa: SLF001
    total = len(plan)
    print(f'\nplan: {total} samples, {params["samples_per_class"]} per class, blocks of {params["block"]}')
    print(f'already on disk: {done}\n')
    if not params['execute']:
        print('DRY RUN: nothing moves and no image is saved. Re-run with -p execute:=true.\n')

    saved = 0
    for n, (label, obj) in enumerate(plan, start=1):
        index = next_index(node._root, label)
        if index > int(params['samples_per_class']):
            continue  # this class is already complete from an earlier session
        position, quat = randomized_pose(reference, level_quat, rng, jitter, params['yaw_deg'], params['tilt_deg'])
        print(f'--- {n}/{total}  class {label}  sample {index}/{params["samples_per_class"]} ---')
        node.move_to(position, quat)

        _prompt(f'    >>> OBJECT: {obj}  ({label}).  ENTER to OPEN the gripper... ')
        node.open_gripper()
        _prompt(f'    place "{obj}" in the jaws. ENTER to CLOSE... ')
        node.close_gripper()
        _prompt('    ENTER to CAPTURE the finger image... ')
        path = node.capture(label, obj, index, position, quat)
        if path is None:
            print('    NOT saved -- fix the stream and press ENTER to retry this sample.')
            input()
            path = node.capture(label, obj, index, position, quat)
        if path is not None:
            saved += 1
            print(f'    saved {path}')
    return saved


def main():
    """Run one collection session."""
    rclpy.init()
    node = TextureCollectNode()
    # The node has to be SPINNING while TF resolves and while frames arrive, so the executor runs
    # in its own thread and the prompts block the main one.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    if node._p['execute']:  # noqa: SLF001
        node.get_logger().warn(f'MOVING THE ARM between samples. {CONSUMER_HINT}')
    try:
        saved = run_session(node)
        print(f'\ndone -- {saved} sample(s) saved under {node._p["root"]}')  # noqa: SLF001
    except (KeyboardInterrupt, EOFError):
        print('\ninterrupted -- re-run to resume where this stopped.')
    except Exception as exc:  # noqa: BLE001 - a CLI tool should print, not traceback
        node.get_logger().error(str(exc))
        return 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
