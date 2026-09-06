"""
Check whether a scene's ORB-SLAM3 mapping pass produced a usable atlas.

Preprocessing step 2 builds one map from the MAPPING session and then localizes every
EPISODE against it, so a bad map costs the whole scene — and it fails quietly: the episodes
localize, they just track badly or not at all, which reads as "SLAM is poor here" rather than
"the map is wrong". This reads the mapping log (and the scene zarr, if step 2 finished) and
says whether the atlas is a complete record of the sweep or only part of one.

The failure being looked for is a **branch**. On losing tracking, ORB-SLAM3 either resets the
active map in place or, once that map is mature enough to be worth keeping (>10 keyframes and
an initialized IMU), parks it and starts a fresh one — ``Tracking::CreateMapInAtlas``. The
Atlas design assumes the two are merged later by place recognition; when that merge does not
happen, the saved atlas holds only part of the sweep, and episodes working in the rest of the
workspace have nothing to relocalize against.

**Sweep coverage is the verdict**, not the branch itself: a branch in the final half second
leaves the surviving map holding everything, while one in the middle splits the sweep in two.
The branch, the per-map keyframe counts and the keyframe density are all reported to explain
the coverage number, but only coverage decides whether the pass should be redone.

Run it after ``pingest pp 2`` — or after the map-building phase alone — so a bad map is caught
before it is worth localizing twenty episodes against:

    uv run python ingest/integration/check_mapping.py recordings/scene_YYYY-MM-DD_hh-mm-ss_XXXX

Exits nonzero when a pass should be redone, so it can gate a batch.
"""

import argparse
import pathlib
import re

#: Fraction of the sweep that must survive into the saved map. This is the metric that
#: matters, and it is already computed for us: SaveTrajectoryCSV writes a frame's pose only
#: when its reference keyframe belongs to the map it selected, marking every other frame lost.
#: So the mapping episode's own tracking ratio *is* the share of the sweep the atlas holds.
#: Clean passes here run 95-98%; a pass whose map fragmented mid-sweep ran 71%.
MIN_SWEEP_COVERAGE = 0.90

#: Keyframes per 1000 fed frames. Advisory only — the count that makes relocalization reliable
#: has not been pinned down. A pass whose episodes went on to track at 94-100% ran 22.7; a
#: texture-starved one whose episodes managed 30-85% ran 12.7.
DENSITY_GOOD = 20.0
DENSITY_THIN = 15.0

_MAP_INIT_RE = re.compile(r'First KF:(\d+); Map init KF:(\d+)')
_MAP_HAS_RE = re.compile(r'Map (\d+) has (\d+) KFs')


def mapping_slam_attrs(scene_dir: pathlib.Path) -> dict | None:
    """
    Read the mapping episode's ``annotations/slam`` attrs, or None if step 2 has not run.

    Absent is the normal case while a mapping pass is still running; the log-derived checks
    still work without it.
    """
    try:
        import zarr

        root = zarr.open(str(scene_dir / 'scene.zarr'), mode='r')
        return dict(root['episode_0']['annotations']['slam'].attrs)
    except Exception:
        return None


def check_scene(scene_dir: pathlib.Path) -> bool:
    """
    Report on one scene's mapping pass; return True when it should be redone.

    Prints the findings rather than returning them: this is a console tool, and the interesting
    output is the per-map keyframe counts, which a caller would only print anyway.
    """
    log = scene_dir / 'slam_logs' / 'mapping_slam.stdout'
    print(f'=== {scene_dir.name} ===')
    if not log.exists():
        print(f'  no mapping log at {log} — has `pingest pp 2` run?')
        return True

    text = log.read_text()
    inits = _MAP_INIT_RE.findall(text)
    maps = [(int(map_id), int(n_kfs)) for map_id, n_kfs in _MAP_HAS_RE.findall(text)]

    if 'maps in the atlas' not in text:
        print('  mapping pass is STILL RUNNING (no final atlas summary yet)')

    print(
        f'  map initialisations: {len(inits)}   '
        f'resets: {text.count("Reseting active map")}   '
        f'track-lost events: {text.count("Track lost")}'
    )
    for first_kf, init_kf in inits:
        print(f'    First KF:{first_kf}; Map init KF:{init_kf}')

    redo = False

    # A map initialized at a nonzero keyframe is a branch. On its own this says nothing about
    # whether the pass is usable: a branch in the last half second of the recording leaves the
    # surviving map holding the entire sweep, while one in the middle splits it in half. What
    # the branch cost is measured below, by coverage — this is reported to explain that number.
    branch_kfs = [int(init_kf) for _, init_kf in inits if int(init_kf) > 0]
    if branch_kfs:
        print(f'  branched: a new map was started at keyframe {min(branch_kfs)} (see coverage below for what it cost).')
    else:
        print('  no branch: the whole sweep stayed in one map.')

    if maps:
        populated = [(map_id, n_kfs) for map_id, n_kfs in maps if n_kfs > 0]
        print(f'  final atlas: {len(maps)} map(s); populated: {populated}')
        if populated and populated[0][0] != 0:
            print(f'    surviving map id is {populated[0][0]}, not 0 — a branch happened.')

    attrs = mapping_slam_attrs(scene_dir)
    if attrs is None:
        print('  coverage: unknown (step 2 has not written the mapping episode yet)')
        return True

    # The decisive number. SaveTrajectoryCSV writes a pose only for frames whose reference
    # keyframe is in the map it selected, so the mapping episode's own tracking ratio is the
    # share of the sweep that made it into the atlas the episodes will localize against.
    coverage = float(attrs.get('tracking_ratio', 0.0))
    print(f'  sweep coverage: {100 * coverage:.1f}% of the mapping pass is in the saved map')
    if coverage < MIN_SWEEP_COVERAGE:
        print(
            f'    below {100 * MIN_SWEEP_COVERAGE:.0f}% — the map is missing part of the '
            'workspace; episodes working there will not localize.'
        )
        redo = True

    n_kfs = max((n_kfs for _, n_kfs in maps), default=0)
    fed = int(attrs.get('n_frames_fed', 0))
    if fed:
        density = 1000.0 * n_kfs / fed
        band = 'thin' if density < DENSITY_THIN else ('ok' if density < DENSITY_GOOD else 'good')
        print(f'  keyframes: {n_kfs} over {fed} fed frames = {density:.1f} per 1000 ({band}, advisory)')

    print(f'  VERDICT: {"REDO THIS MAPPING PASS" if redo else "looks clean"}\n')
    return redo


def main() -> int:
    """Check every scene named on the command line; return a shell exit code."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        'scene_dirs',
        nargs='+',
        type=pathlib.Path,
        help='Scene directories containing slam_logs/mapping_slam.stdout.',
    )
    args = parser.parse_args()
    return 1 if any([check_scene(d.resolve()) for d in args.scene_dirs]) else 0


if __name__ == '__main__':
    raise SystemExit(main())
