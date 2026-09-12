#!/usr/bin/env python3
"""
Water-bottle shake: level the tool, then shake it vertically a fixed number of times.

Built to produce a labelled audio corpus. The arm holds a bottle, shakes it the SAME way every
run, and a bag records the contact mic, the finger camera and the GoPro while it does. The water
level is the label -- it comes from how the bottle was filled, not from anything measured here --
so the only thing this node owes the dataset is that every trial move identically. Everything
below is in service of that: a fixed waypoint grid, a closed-form height profile, and a starting
orientation derived from the robot rather than from wherever the operator left it.

**Levelling.** ``polyumi_tcp`` is in GoPro-optical axes (x right, y down, z forward, and z is the
approach axis -- see nuc/tcp_calib.py). "Perfectly horizontal" here means:

    z' (approach)     horizontal, pointing the way it already pointed
    y' (optical down) straight down, along world -Z
    x' = y' x z'      horizontal

i.e. the tool's yaw is kept and its roll and pitch are zeroed. The bottle ends up held level,
facing the same direction as before, which is the least surprising thing to do to a pose the
operator chose. The current approach axis is projected onto the world XY plane to get that yaw,
so a tool already pointing straight up or down has no defined yaw and the node refuses rather
than picking one.

**The shake** is a raised cosine in world Z about the levelled start:

    z(t) = z0 + A * (1 - cos(2*pi*t/T)) / 2

One period is one shake: up to +A at T/2 and back to z0 at T. It never goes BELOW z0, so the
motion cannot drive the bottle into the table even if the start pose is low. Position x and y and
the orientation are held fixed throughout -- only height changes.

Usage (laptop, after `source setup_franka_env.sh`):

    # 1. NUC: bringup + inference, arm execution on.
    ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true

    # 2. Put the arm somewhere roomy with the bottle gripped, then DRY RUN (default): nothing
    #    moves, and the whole commanded path shows up on /polyumi/target_poses_preview.
    ros2 run polyumi_ros2 water_shake

    # 3. Watch the preview in Foxglove. When it looks right, execute:
    ros2 run polyumi_ros2 water_shake --ros-args -p execute:=true

    # 4. Record a trial (separate shell), then run step 3 again:
    ros2_ws/src/polyumi_ros2/scripts/record_eval_trial.sh water_half_full

**Do not run this while policy_client_node is running.** Both publish to
/polyumi/target_poses_traj and the controller splices whichever chunk arrives last.
"""

import math
import sys

from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from polyumi_ros2.target_chunk import CONSUMER_HINT, TargetChunkPublisher
from tf2_ros import Buffer, TransformListener

#: Below this, the approach axis is too close to vertical for its projection onto the world XY
#: plane to define a yaw. 0.1 rad of tilt off vertical still leaves a 0.0998 horizontal component,
#: so this only rejects a tool genuinely pointing up or down.
MIN_HORIZONTAL_APPROACH = 0.1

PREVIEW_TOPIC = '/polyumi/target_poses_preview'


def level_rotation(approach_world: np.ndarray) -> Rotation:
    """
    Build the levelled TCP orientation that keeps `approach_world`'s yaw and zeroes roll/pitch.

    :param approach_world: the TCP's current z axis (approach) in the base frame, any length.
    :returns: the levelled rotation, as base_R_tcp.
    :raises ValueError: if the approach axis is too near vertical to define a yaw.

    The returned frame is right-handed by construction: x' = y' x z' with y' straight down.
    """
    approach = np.asarray(approach_world, dtype=float)
    horizontal = float(np.hypot(approach[0], approach[1]))
    if horizontal < MIN_HORIZONTAL_APPROACH:
        raise ValueError(
            f'approach axis is {horizontal:.3f} from vertical in the XY plane (limit '
            f'{MIN_HORIZONTAL_APPROACH}); it points up or down, so "keep the yaw" means nothing. '
            'Rotate the tool to point roughly sideways first.'
        )
    yaw = math.atan2(approach[1], approach[0])
    z_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))


def shake_heights(amplitude_m: float, n_shakes: int, period_s: float, dt: float) -> np.ndarray:
    """
    Height offsets above the start, one per waypoint, for `n_shakes` raised-cosine shakes.

    :returns: offsets in metres, starting and ending at 0.0, never negative.

    The grid is closed-form rather than accumulated so every trial lands on the same instants:
    the classifier's whole premise is that the only thing differing between runs is the water.
    """
    if amplitude_m <= 0 or n_shakes < 1 or period_s <= 0 or dt <= 0:
        raise ValueError('amplitude_m, period_s and dt must be > 0 and n_shakes >= 1')
    total_s = n_shakes * period_s
    n_steps = int(round(total_s / dt))
    t = np.arange(n_steps + 1) * dt
    return amplitude_m * (1.0 - np.cos(2.0 * math.pi * t / period_s)) / 2.0


def slerp_quats(q_from: np.ndarray, q_to: np.ndarray, n: int) -> np.ndarray:
    """Interpolate `n` quaternions from `q_from` to `q_to` inclusive, as xyzw rows."""
    if n < 2:
        return np.asarray([q_to], dtype=float)
    key = Rotation.from_quat(np.vstack([q_from, q_to]))
    from scipy.spatial.transform import Slerp

    return Slerp([0.0, 1.0], key)(np.linspace(0.0, 1.0, n)).as_quat()


def _pose(position: np.ndarray, quat: np.ndarray) -> Pose:
    """Build a geometry_msgs Pose from a 3-vector and an xyzw quaternion."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in position)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in quat)
    return p


class WaterShakeNode(Node):
    """Level the TCP, then shake it vertically N times, as one absolutely-timed chunk."""

    def __init__(self, **kwargs):
        """Declare parameters and create the chunk and preview publishers."""
        super().__init__('water_shake', **kwargs)
        self.declare_parameter('amplitude_m', 0.08)
        self.declare_parameter('period_s', 1.0)
        self.declare_parameter('n_shakes', 5)
        self.declare_parameter('waypoint_dt', 0.05)
        self.declare_parameter('level_time_s', 2.0)
        self.declare_parameter('settle_s', 0.5)
        # Default false, like policy_client_node's execute_motion: running this by accident must
        # not move an arm holding a full bottle. The preview is published either way.
        self.declare_parameter('execute', False)
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('eef_frame', 'polyumi_tcp')

        self._p = {
            k: self.get_parameter(k).value
            for k in (
                'amplitude_m',
                'period_s',
                'n_shakes',
                'waypoint_dt',
                'level_time_s',
                'settle_s',
                'execute',
                'base_frame',
                'eef_frame',
            )
        }

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._preview = self.create_publisher(PoseArray, PREVIEW_TOPIC, 10)
        self._pub = (
            TargetChunkPublisher(self, frame_id=self._p['base_frame'], joint_name=self._p['eef_frame'])
            if self._p['execute']
            else None
        )

    def lookup_tcp(self, timeout_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, xyzw quaternion) of the TCP in the base frame."""
        tf = self._tf_buffer.lookup_transform(
            self._p['base_frame'],
            self._p['eef_frame'],
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=timeout_s),
        )
        t, r = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w])

    def build_poses(self, position: np.ndarray, quat: np.ndarray) -> list[Pose]:
        """
        Build the full commanded path: level in place, settle, then shake.

        Position is held at `position` for the levelling and the settle; only the shake moves it,
        and only in world Z.
        """
        dt = self._p['waypoint_dt']
        level_quat = level_rotation(Rotation.from_quat(quat).as_matrix()[:, 2]).as_quat()

        n_level = max(2, int(round(self._p['level_time_s'] / dt)))
        poses = [_pose(position, q) for q in slerp_quats(quat, level_quat, n_level)]
        poses += [_pose(position, level_quat)] * max(0, int(round(self._p['settle_s'] / dt)))

        for dz in shake_heights(self._p['amplitude_m'], int(self._p['n_shakes']), self._p['period_s'], dt):
            poses.append(_pose(position + np.array([0.0, 0.0, dz]), level_quat))
        return poses

    def run_once(self) -> int:
        """Look up the TCP, build the path, publish it. Returns the waypoint count."""
        position, quat = self.lookup_tcp()
        poses = self.build_poses(position, quat)

        preview = PoseArray()
        preview.header.frame_id = self._p['base_frame']
        preview.header.stamp = self.get_clock().now().to_msg()
        preview.poses = poses
        self._preview.publish(preview)

        span = len(poses) * self._p['waypoint_dt']
        if self._pub is None:
            self.get_logger().warn(
                f'DRY RUN — {len(poses)} waypoints ({span:.1f}s) on {PREVIEW_TOPIC}. '
                'Nothing was commanded. Re-run with -p execute:=true to move the arm.'
            )
        else:
            self._pub.publish(poses, dt=self._p['waypoint_dt'])
            self.get_logger().warn(
                f'MOVING THE ARM — {len(poses)} waypoints ({span:.1f}s), '
                f'{self._p["n_shakes"]} shakes of {self._p["amplitude_m"] * 100:.0f} cm. '
                f'{CONSUMER_HINT}'
            )
        return len(poses)


def main():
    """Publish one shake sequence and exit."""
    rclpy.init()
    node = WaterShakeNode()
    try:
        # Spin briefly so the TF listener fills before the lookup.
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
        node.run_once()
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
    except Exception as exc:  # noqa: BLE001 - a CLI tool should print, not traceback
        node.get_logger().error(str(exc))
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
