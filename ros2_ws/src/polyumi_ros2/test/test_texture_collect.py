"""
Tests for texture_collect's planning, pose randomisation and on-disk layout.

These are the parts a bad run cannot recover from: a plan that is not interleaved bakes a
collection-order confound into 250 samples, a resume that miscounts overwrites a class, and a
pose draw outside its bounds sends the arm somewhere the operator did not agree to. All are pure
functions, so none of this needs a robot, a node, or rclpy.
"""

# ruff: noqa: D103  - test functions are self-describing via names + inline comments

import json
import pathlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from polyumi_ros2.texture_collect import (
    build_plan,
    load_classes,
    manifest_row,
    next_index,
    randomized_pose,
    sample_path,
)

CLASSES = {'metal': ['ruler', 'can'], 'wood': ['block'], 'foam': ['sponge']}


def _rng():
    return np.random.default_rng(0)


def test_plan_gives_every_class_the_same_number_of_samples():
    plan = build_plan(CLASSES, n_per_class=10, block=5, rng=_rng())

    assert len(plan) == 30
    for label in CLASSES:
        assert sum(1 for c, _ in plan if c == label) == 10


def test_plan_interleaves_classes_in_blocks_rather_than_one_after_another():
    """Collected one class at a time, any drift over the session lines up with the label."""
    plan = build_plan(CLASSES, n_per_class=10, block=5, rng=_rng())
    labels = [c for c, _ in plan]

    assert labels[:5] == ['metal'] * 5
    assert labels[5:10] == ['wood'] * 5
    assert labels[10:15] == ['foam'] * 5
    assert labels[15:20] == ['metal'] * 5  # round two, not the rest of round one


def test_plan_last_round_takes_the_remainder():
    plan = build_plan(CLASSES, n_per_class=12, block=5, rng=_rng())

    assert [c for c, _ in plan].count('metal') == 12
    assert len(plan) == 36


def test_plan_draws_objects_from_the_right_class():
    plan = build_plan(CLASSES, n_per_class=20, block=4, rng=_rng())

    for label, obj in plan:
        assert obj in CLASSES[label]
    # With 20 draws from two objects, both should appear -- a fixed object per class would be a
    # dataset of one object, which the class label would then misdescribe.
    assert {obj for label, obj in plan if label == 'metal'} == {'ruler', 'can'}


@pytest.mark.parametrize('bad', [0, -1])
def test_plan_rejects_nonsense_counts(bad):
    with pytest.raises(ValueError, match='>= 1'):
        build_plan(CLASSES, n_per_class=bad, block=5, rng=_rng())


def test_pose_stays_inside_the_position_and_angle_bounds():
    reference = np.array([0.4, 0.0, 0.3])
    level = Rotation.identity().as_quat()
    jitter = np.array([0.04, 0.04, 0.03])
    rng = _rng()

    for _ in range(200):
        position, quat = randomized_pose(reference, level, rng, jitter, yaw_deg=25.0, tilt_deg=8.0)
        assert np.all(np.abs(position - reference) <= jitter + 1e-12)
        # Total angle off the levelled orientation cannot exceed yaw + tilt.
        angle = Rotation.from_quat(quat).magnitude()
        assert angle <= np.radians(25.0 + 8.0) + 1e-9


def test_pose_actually_varies():
    """A pose that does not move leaves every sample photographing one scene."""
    reference = np.zeros(3)
    level = Rotation.identity().as_quat()
    rng = _rng()

    draws = [randomized_pose(reference, level, rng, np.full(3, 0.04), 25.0, 8.0) for _ in range(10)]

    assert len({tuple(np.round(p, 6)) for p, _ in draws}) == 10
    assert len({tuple(np.round(q, 6)) for _, q in draws}) == 10


def test_next_index_starts_at_one_and_resumes_past_existing_samples(tmp_path):
    assert next_index(tmp_path, 'metal') == 1

    (tmp_path / 'metal').mkdir()
    for n in (1, 2, 7):  # 7 with a gap: a deleted bad sample must not be written over
        sample_path(tmp_path, 'metal', 'ruler', n).touch()

    assert next_index(tmp_path, 'metal') == 8


def test_sample_path_is_one_directory_per_class():
    """Torchvision's ImageFolder takes the class from the directory, so the tree IS the labelling."""
    path = sample_path(pathlib.Path('/data/tex'), 'metal', 'ruler', 3)

    assert path == pathlib.Path('/data/tex/metal/metal_ruler_0003.png')


def test_manifest_row_records_what_the_image_cannot(tmp_path):
    path = sample_path(tmp_path, 'metal', 'ruler', 3)

    row = manifest_row(
        'metal',
        'ruler',
        3,
        path,
        np.array([0.4, 0.1, 0.3]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        0.031,
        {'x_min': 170, 'x_max': None, 'y_min': 0, 'y_max': None},
        (648, 982, 3),
    )

    assert row['class'] == 'metal' and row['object'] == 'ruler'
    assert row['file'] == 'metal/metal_ruler_0003.png'  # relative, so the root can move
    assert row['grip_width_m'] == pytest.approx(0.031)
    assert row['tcp_position_xyz'] == [0.4, 0.1, 0.3]
    assert json.loads(json.dumps(row))  # must survive a JSONL round trip


def test_load_classes_reads_the_plan(tmp_path):
    path = tmp_path / 'classes.json'
    path.write_text(json.dumps(CLASSES))

    assert load_classes(path) == CLASSES


@pytest.mark.parametrize(
    'payload',
    [
        {'metal': []},  # a class with no objects cannot be sampled
        {'met al': ['ruler']},  # becomes a directory name
        {'metal': ['rul/er']},  # becomes a file name
        {},
    ],
)
def test_load_classes_rejects_names_a_path_cannot_hold(tmp_path, payload):
    path = tmp_path / 'classes.json'
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_classes(path)
