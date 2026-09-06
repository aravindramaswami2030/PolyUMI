#!/usr/bin/env python3
"""
Report SLAM tracking quality per episode, and diff two runs against each other.

Built for evaluating a change to how the map is made (``accumulate_map``, say): capture a
baseline before the change, re-run preprocessing, then diff.  Reads only what step 2 already
writes to ``annotations/slam``, so it costs nothing and needs no decoding.

    # before changing anything
    uv run python ingest/integration/slam_report.py recordings/scene_X --save before.json

    # after re-running step 2 (and steps 3-6)
    uv run python ingest/integration/slam_report.py recordings/scene_X --compare before.json

``tracking_ratio`` is the headline number but not the whole story: DP export additionally cuts
on pose jumps, so frames SLAM called tracked can still be dropped downstream.  Treat a
tracking gain as promising rather than proven until the export minutes agree.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import zarr

#: Attrs worth reporting, in display order.
_FIELDS = ('tracking_ratio', 'n_relocalization_events', 'n_frames_fed', 'n_frames_fed_tracked')


def _scene_zarr(scene_dir: pathlib.Path) -> pathlib.Path:
    """Return the pzarr store inside a scene directory."""
    direct = scene_dir / 'scene.zarr'
    if direct.exists():
        return direct
    candidates = sorted(scene_dir.glob('*.zarr'))
    if not candidates:
        raise FileNotFoundError(f'no .zarr store under {scene_dir}')
    return candidates[0]


def collect(scene_dir: pathlib.Path) -> dict:
    """Read per-episode SLAM annotations out of a scene's store."""
    root = zarr.open_group(str(_scene_zarr(scene_dir)), mode='r')
    episodes = {}
    for key in sorted(root):
        if not key.startswith('episode'):
            continue
        grp = root[key]
        if 'annotations/slam' not in grp:
            continue
        attrs = dict(grp['annotations/slam'].attrs)
        episodes[key] = {
            'session_type': grp.attrs.get('session_type', '?'),
            'accumulated_map': attrs.get('accumulated_map', False),
            **{f: attrs.get(f) for f in _FIELDS},
        }
    return {'scene': scene_dir.name, 'episodes': episodes}


def _fmt_ratio(value: float | None) -> str:
    return '   n/a' if value is None else f'{100.0 * value:5.1f}%'


def _print_report(report: dict) -> None:
    episodes = report['episodes']
    print(f'== {report["scene"]} ==')
    if not episodes:
        print('   (no episode carries annotations/slam -- has step 2 run?)')
        return
    for key, ep in episodes.items():
        print(
            f'   {key:12s} {ep["session_type"]:8s} tracking={_fmt_ratio(ep["tracking_ratio"])}'
            f'  relocs={ep["n_relocalization_events"]}'
            f'  fed={ep["n_frames_fed_tracked"]}/{ep["n_frames_fed"]}'
            f'  accumulated={ep["accumulated_map"]}'
        )
    ratios = [e['tracking_ratio'] for e in episodes.values() if e['tracking_ratio'] is not None]
    relocs = [e['n_relocalization_events'] for e in episodes.values() if e['n_relocalization_events'] is not None]
    if ratios:
        print(f'   -- mean tracking {_fmt_ratio(sum(ratios) / len(ratios))}, {sum(relocs)} relocalizations total')


def _print_comparison(before: dict, after: dict) -> None:
    print(f'== {after["scene"]}: before -> after ==')
    keys = sorted(set(before['episodes']) | set(after['episodes']))
    d_ratios = []
    for key in keys:
        b = before['episodes'].get(key)
        a = after['episodes'].get(key)
        if b is None or a is None:
            print(f'   {key:12s} only in {"after" if b is None else "before"}')
            continue
        br, ar = b['tracking_ratio'], a['tracking_ratio']
        delta = '' if br is None or ar is None else f'  ({100.0 * (ar - br):+5.1f} pts)'
        if br is not None and ar is not None:
            d_ratios.append(ar - br)
        print(
            f'   {key:12s} tracking {_fmt_ratio(br)} -> {_fmt_ratio(ar)}{delta}'
            f'   relocs {b["n_relocalization_events"]} -> {a["n_relocalization_events"]}'
        )
    if d_ratios:
        mean = sum(d_ratios) / len(d_ratios)
        verdict = 'BETTER' if mean > 0.005 else ('WORSE' if mean < -0.005 else 'NO CHANGE')
        print(f'   -- mean tracking change {100.0 * mean:+.1f} pts: {verdict}')
        print('   -- tracking is not the whole story; confirm with usable DP export minutes.')


def main() -> int:
    """Report or diff SLAM tracking quality for a scene."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene_dir', type=pathlib.Path, help='scene directory (contains scene.zarr)')
    parser.add_argument('--save', type=pathlib.Path, help='write this run to JSON, as a baseline to diff later')
    parser.add_argument('--compare', type=pathlib.Path, help='diff the current store against a saved baseline')
    args = parser.parse_args()

    if not args.scene_dir.is_dir():
        print(f'error: {args.scene_dir} is not a directory', file=sys.stderr)
        return 2

    report = collect(args.scene_dir)

    if args.compare:
        baseline = json.loads(args.compare.read_text())
        _print_comparison(baseline, report)
    else:
        _print_report(report)

    if args.save:
        args.save.write_text(json.dumps(report, indent=2))
        print(f'   baseline written to {args.save}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
