"""
Tests for the water-bottle shake controller's geometry and height profile.

The classifier's premise is that every trial moves identically, so the things worth pinning are
the ones that would silently differ between runs: the levelled orientation the shake starts from,
and the exact height grid. Both are pure functions here, so none of this needs a robot.

    bash -c 'unset VIRTUAL_ENV; cd ros2_ws && source /opt/ros/kilted/setup.bash \
      && source install/setup.bash && cd src/polyumi_ros2 \
      && /usr/bin/python3 -m pytest test/test_water_shake.py -q'
"""

import math

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.parameter import Parameter

from polyumi_ros2.water_shake import (
    WaterShakeNode,
    level_rotation,
    shake_heights,
    slerp_quats,
)


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; the node here never runs under a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def make_node():
    """Build a WaterShakeNode with parameter overrides; its publishers are real but unused."""
    nodes = []

    def _make(**overrides):
        params = [Parameter(k, value=v) for k, v in overrides.items()]
        node = WaterShakeNode(parameter_overrides=params)
        node.get_logger = MagicMock()
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node.destroy_node()


def _axes(rot: Rotation):
    """Return the (x, y, z) column axes of a rotation, as base-frame vectors."""
    m = rot.as_matrix()
    return m[:, 0], m[:, 1], m[:, 2]


# ----------------------------------------------------------------------
# Levelling
# ----------------------------------------------------------------------


@pytest.mark.parametrize('yaw_deg', [0.0, 37.0, 90.0, -125.0, 179.0])
def test_levelled_approach_is_horizontal_and_keeps_yaw(yaw_deg):
    """The whole point: roll and pitch go to zero, the direction the tool faces does not."""
    yaw = math.radians(yaw_deg)
    # An approach axis with the requested yaw and an arbitrary, wrong, pitch.
    approach = np.array([math.cos(yaw), math.sin(yaw), 0.4])
    _, _, z = _axes(level_rotation(approach))

    assert z[2] == pytest.approx(0.0, abs=1e-12), 'approach axis must end up horizontal'
    assert math.atan2(z[1], z[0]) == pytest.approx(yaw, abs=1e-9), 'yaw must be preserved'


def test_levelled_optical_down_points_at_world_down():
    """The y axis is 'down' in GoPro-optical axes, so levelling must aim it at world -Z."""
    _, y, _ = _axes(level_rotation(np.array([1.0, 0.0, -0.3])))
    assert y == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_levelled_frame_is_right_handed_and_orthonormal():
    """A left-handed or skewed frame would be a valid-looking quaternion commanding a mirror."""
    m = level_rotation(np.array([0.3, -0.9, 0.2])).as_matrix()
    assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-12)
    assert m.T @ m == pytest.approx(np.eye(3), abs=1e-12)


def test_levelling_is_idempotent():
    """Re-levelling an already level tool must not nudge it, or repeat trials would drift."""
    first = level_rotation(np.array([1.0, 1.0, 0.0]))
    second = level_rotation(_axes(first)[2])
    assert second.as_matrix() == pytest.approx(first.as_matrix(), abs=1e-12)


@pytest.mark.parametrize('approach', [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.05, 0.05, 1.0]])
def test_near_vertical_approach_is_refused(approach):
    """Straight up or down has no yaw to keep; picking one silently would be a surprise move."""
    with pytest.raises(ValueError, match='up or down'):
        level_rotation(np.array(approach))


# ----------------------------------------------------------------------
# Height profile
# ----------------------------------------------------------------------


def test_shake_starts_and_ends_at_the_start_height():
    """The arm must finish where it began, so trials can be run back to back without a reset."""
    h = shake_heights(amplitude_m=0.08, n_shakes=5, period_s=1.0, dt=0.05)
    assert h[0] == pytest.approx(0.0, abs=1e-12)
    assert h[-1] == pytest.approx(0.0, abs=1e-9)


def test_shake_never_goes_below_the_start():
    """A raised cosine, not a sine: dipping below the start could drive the bottle into the table."""
    h = shake_heights(amplitude_m=0.08, n_shakes=5, period_s=1.0, dt=0.05)
    assert h.min() >= -1e-12


def test_shake_reaches_full_amplitude_once_per_shake():
    """Five shakes must be five peaks -- that is the number the label set is built around."""
    h = shake_heights(amplitude_m=0.08, n_shakes=5, period_s=1.0, dt=0.01)
    peaks = [i for i in range(1, len(h) - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1]]
    assert len(peaks) == 5
    assert h.max() == pytest.approx(0.08, abs=1e-9)


def test_shake_grid_is_deterministic_across_calls():
    """Identical motion every trial is the experiment's one requirement."""
    a = shake_heights(0.08, 5, 1.0, 0.05)
    b = shake_heights(0.08, 5, 1.0, 0.05)
    assert a == pytest.approx(b, abs=0.0)


def test_shake_duration_matches_n_shakes_times_period():
    """Waypoint count sets the bag length, which is what the classifier windows against."""
    h = shake_heights(0.08, n_shakes=5, period_s=1.0, dt=0.05)
    assert len(h) - 1 == pytest.approx(5 * 1.0 / 0.05)


@pytest.mark.parametrize(
    'bad',
    [
        dict(amplitude_m=0.0, n_shakes=5, period_s=1.0, dt=0.05),
        dict(amplitude_m=0.08, n_shakes=0, period_s=1.0, dt=0.05),
        dict(amplitude_m=0.08, n_shakes=5, period_s=0.0, dt=0.05),
        dict(amplitude_m=0.08, n_shakes=5, period_s=1.0, dt=0.0),
    ],
)
def test_degenerate_shake_parameters_are_refused(bad):
    """Silently producing a one-waypoint 'shake' would look like a working but useless trial."""
    with pytest.raises(ValueError):
        shake_heights(**bad)


# ----------------------------------------------------------------------
# Orientation ramp
# ----------------------------------------------------------------------


def test_slerp_hits_both_endpoints():
    """The ramp must actually arrive at the levelled orientation, not near it."""
    q0 = Rotation.from_euler('xyz', [0.2, -0.3, 0.5]).as_quat()
    q1 = Rotation.from_euler('xyz', [0.0, 0.0, 0.5]).as_quat()
    out = slerp_quats(q0, q1, 10)
    assert len(out) == 10
    assert Rotation.from_quat(out[0]).approx_equal(Rotation.from_quat(q0), atol=1e-9)
    assert Rotation.from_quat(out[-1]).approx_equal(Rotation.from_quat(q1), atol=1e-9)


# ----------------------------------------------------------------------
# Gripper squeeze
# ----------------------------------------------------------------------


def test_squeeze_commands_the_measured_width_minus_the_squeeze(make_node):
    """The target is derived from what the jaws report, not configured, so it fits today's bottle."""
    node = make_node(grip_squeeze_m=0.002)
    node._grip_width = 0.0450
    published = []
    node._grip_pub = MagicMock(publish=lambda m: published.append(m))

    target = node.squeeze()

    assert target == pytest.approx(0.0430)
    assert published[0].joint_names == ['fr3_gripper_width']
    assert published[0].points[0].positions == pytest.approx([0.0430])


def test_squeeze_never_commands_a_negative_width(make_node):
    """A bottle thinner than the squeeze would otherwise ask the jaws past their own zero."""
    node = make_node(grip_squeeze_m=0.05)
    node._grip_width = 0.01
    node._grip_pub = MagicMock()

    assert node.squeeze() == pytest.approx(0.0)


def test_squeeze_gives_up_without_gripper_state_rather_than_hanging(make_node):
    """No gripper feedback must not abort a trial mid-collection -- warn and carry on."""
    node = make_node()
    node._grip_pub = MagicMock()

    assert node.squeeze(timeout_s=0.3) is None
    node._grip_pub.publish.assert_not_called()


def test_squeeze_is_skipped_on_a_dry_run(make_node):
    """A preview must not move the fingers any more than it moves the arm."""
    node = make_node(execute=False)
    node._grip_width = 0.045
    node._grip_pub = MagicMock()
    # A tool pointing sideways: identity would aim the approach axis straight up, which
    # level_rotation refuses by design.
    sideways = Rotation.from_euler('y', math.pi / 2).as_quat()
    node.lookup_tcp = lambda **_: (np.zeros(3), sideways)
    node._preview = MagicMock()

    node.run_once()

    node._grip_pub.publish.assert_not_called()


# ----------------------------------------------------------------------
# Shake axis and waveform
# ----------------------------------------------------------------------


def test_symmetric_shake_is_centred_on_the_start():
    """Sideways wants the start at the centre of travel, not at one end -- twice the excursion."""
    h = shake_heights(0.05, n_shakes=5, period_s=0.3, dt=0.01, symmetric=True)
    assert h.min() == pytest.approx(-0.05, abs=1e-3)
    assert h.max() == pytest.approx(0.05, abs=1e-3)
    assert h[0] == pytest.approx(0.0, abs=1e-12)
    assert h[-1] == pytest.approx(0.0, abs=1e-9)


def test_one_sided_shake_still_never_goes_negative():
    """The vertical default must keep its no-dig-into-the-table property."""
    h = shake_heights(0.05, n_shakes=5, period_s=0.3, dt=0.01, symmetric=False)
    assert h.min() >= -1e-12


def test_shake_displaces_along_the_requested_axis(make_node):
    """A sideways axis must move sideways and not vertically -- that is the whole change."""
    # waypoint_dt matters here: at the 0.05 default a 0.3 s period samples only 6 points per
    # cycle and never lands on the sine's peak, so the travel comes out 2A*sin(60deg).
    node = make_node(shake_axis=[1.0, 0.0, 0.0], symmetric=True, amplitude_m=0.05,
                     period_s=0.3, waypoint_dt=0.01)
    sideways = Rotation.from_euler('y', math.pi / 2).as_quat()
    poses = node.build_poses(np.array([0.5, 0.0, 0.4]), sideways)
    xyz = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])

    assert xyz[:, 0].ptp() == pytest.approx(0.10, abs=1e-3), 'should travel 2A along x'
    assert xyz[:, 2].ptp() == pytest.approx(0.0, abs=1e-9), 'must not move vertically'


def test_shake_axis_is_normalised(make_node):
    """An unnormalised axis must not silently scale the amplitude."""
    node = make_node(shake_axis=[0.0, 3.0, 0.0], amplitude_m=0.05, period_s=0.3)
    assert float(np.linalg.norm(node._axis)) == pytest.approx(1.0)
    assert node._axis == pytest.approx([0.0, 1.0, 0.0])


@pytest.mark.parametrize('bad', [[0.0, 0.0, 0.0], [1.0, 0.0], [float('nan'), 0.0, 1.0]])
def test_degenerate_shake_axis_is_refused(bad, make_node):
    """A zero or malformed axis would produce a shake that does not move."""
    with pytest.raises(ValueError, match='shake_axis'):
        make_node(shake_axis=bad)
