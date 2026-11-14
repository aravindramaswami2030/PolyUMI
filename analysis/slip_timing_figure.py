"""
Command-locked timing of the slip-control channels, for the CoRL reviewer response.

The task is controlled slip: the object starts at a set angle and the policy brings it to 90
degrees by modulating grip width, deliberately letting it slide a little at a time. So every
commanded release is the policy performing the task, not a confound to be removed.

Events are locked to the **commanded release onset** -- the first upward crossing of the
reconstructed commanded delta. Anchoring on the command matters: an earlier version anchored on
the gripper-width excursion itself, which forces the width peak to land at or after t=0 while
leaving tactile free to peak anywhere, biasing the comparison toward finding tactile early.
The command channel is independent of both channels being compared, so it carries no such bias.

Four channels are shown:

  tactile       mean |frame - prev frame| over the fingertip crop -- a vision-based tactile
                sensor, so it legitimately sees the fingers as well as the object
  width resid   gripper width minus its own 3 s rolling median (excursions are ~1 mm on an
                81.2 mm stroke, so the raw signal shows nothing)
  |d width/dt|  the control for the above: what a camera watching only the fingers would
                register. If tactile merely tracked finger kinematics it would peak here.
  audio         piezo contact-mic envelope

Timestamps use bag **log time** (host receive clock) for every channel. The Pi stamps capture
instants in its own clock and those arrive ~145 ms later; mixing a Pi header stamp against a
host-stamped width would read that transport delay as a tactile lead. Log time also answers the
question that matters for control -- when the information reaches the policy, not when the
photon landed -- and it is the conservative choice, since it penalises the Pi streams.

Consumes the .npz written by extract_slip_rollout.py.
"""

from __future__ import annotations

import argparse
import pathlib
from math import comb

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  - backend must be selected first

GRID = np.arange(-1.0, 2.01, 0.05)
STROKE_M = 0.0812  # caliper-measured PolyUMI finger stroke
CMD_THRESH_M = 0.0003  # smallest commanded opening counted as a release
DEFAULT_OBS_AGE_S = 0.34

# Categorical slots of the validated reference palette. Each channel owns a titled panel, so
# identity never rests on hue alone.
CH = (
    ('tac', 'Tactile activity', 'z', '#1baf7a'),
    ('relw', 'Gripper width residual', 'mm', '#2a78d6'),
    ('wvel', 'Gripper speed |d width/dt|', 'z', '#6250d6'),
    ('aud', 'Contact-mic envelope', 'z', '#e34948'),
)
INK, INK2, MUTED, GRIDC, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'


def rolling_median(x: np.ndarray, t: np.ndarray, win: float = 3.0) -> np.ndarray:
    """Median of x over a centred window of `win` seconds, evaluated at every sample."""
    return np.array([np.median(x[(t >= ti - win / 2) & (t <= ti + win / 2)]) for ti in t])


def robust_z(v: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """z-score v against the median/MAD of `baseline`, so trials of different gain compare."""
    med = np.median(baseline)
    return (v - med) / (1.4826 * np.median(np.abs(baseline - med)) + 1e-12)


def load_events(store: pathlib.Path) -> tuple[list[dict[str, np.ndarray]], tuple[int, int]]:
    """Collect every commanded-release event across the rollouts under `store`."""
    events: list[dict[str, np.ndarray]] = []
    used = total = 0

    for path in sorted(store.glob('*_full.npz')):
        total += 1
        d = np.load(path)
        if d['width_v'].size < 100:  # aborted within ~2 s: too short for a 3 s rolling median
            continue
        used += 1

        wt, wv = d['width_t'].astype(float), d['width_v'].astype(float)
        ct, cv = d['cmd_t'].astype(float), d['cmd_v'].astype(float)
        tt, ta = d['tactile_t'].astype(float), d['tactile_act'].astype(float)
        at, pcm = d['audio_t'].astype(float), d['audio_pcm']

        t0 = min(wt.min(), at.min())
        wts, cts, tts = (wt - t0) / 1e9, (ct - t0) / 1e9, (tt - t0) / 1e9

        relw = wv - rolling_median(wv, wts)
        wvel = np.abs(np.gradient(wv, wts))

        # Under `action_gripper_repr: relative` the server adds the delta to the base width the
        # observation carried, so the base is the measurement as it was obs_age ago.
        age = (
            np.interp(ct, d['obsage_t'].astype(float), d['obsage_v'])
            if d['obsage_t'].size
            else np.full(ct.size, DEFAULT_OBS_AGE_S)
        )
        pred = cv - np.interp(ct - age * 1e9, wt, wv)

        crossings = np.flatnonzero(np.diff((pred > CMD_THRESH_M).astype(int)) == 1) + 1
        onsets: list[float] = []
        for o in cts[crossings]:
            if not onsets or o - onsets[-1] > 2.0:  # one event per release, not per wobble
                onsets.append(float(o))

        piezo = pcm[0::2].astype(np.float32) / 32768.0  # channel 0 is the piezo contact mic
        duration = (at.max() - at.min()) / 1e9
        sr = piezo.size / (duration + 1e-9)
        block = max(int(sr * 0.05), 1)
        nblocks = max(piezo.size // block, 1)
        env = np.sqrt((piezo[: nblocks * block].reshape(nblocks, block) ** 2).mean(axis=1))
        env_t = (at.min() - t0) / 1e9 + np.arange(nblocks) * 0.05

        for o in onsets:
            if o + GRID[-1] > wts.max() or o + GRID[0] < wts.min():
                continue
            events.append(
                {
                    'tac': robust_z(np.interp(o + GRID, tts, ta), ta),
                    'relw': np.interp(o + GRID, wts, relw) * 1000,
                    'wvel': robust_z(np.interp(o + GRID, wts, wvel), wvel),
                    'aud': robust_z(np.interp(o + GRID, env_t, env), env),
                }
            )
    return events, (used, total)


def peak_stats(traces: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean trace, its bootstrap 95% band, and the peak latency with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(traces)
    mean = traces.mean(axis=0)
    boots = np.array([traces[rng.integers(0, n, n)].mean(axis=0) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    i = int(np.argmax(np.abs(mean)))
    pk_boot = GRID[np.argmax(np.abs(boots), axis=1)]
    return mean, lo, hi, GRID[i], mean[i], np.percentile(pk_boot, [2.5, 97.5])


def sign_test(traces_a: np.ndarray, traces_b: np.ndarray) -> tuple[float, int, int, float]:
    """Per-event peak lead of a over b, with an exact two-sided-equivalent sign test."""
    lead = np.array([GRID[np.argmax(a)] - GRID[np.argmax(b)] for a, b in zip(traces_a, traces_b)])
    n = len(lead)
    k = int(np.sum(lead < 0))
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return float(np.median(lead)), k, n, p


def style(ax) -> None:
    """Recessive axes: no top/right spines, muted grid, text in ink tokens."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRIDC)
    ax.grid(axis='y', color=GRIDC, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def draw(events, counts, out: pathlib.Path) -> None:
    """Render the stacked channel traces and the peak-latency summary."""
    traces = {k: np.array([e[k] for e in events]) for k, *_ in CH}
    n = len(events)

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 9.0), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    summary = []

    for ax, (key, label, unit, color) in zip(axes, CH):
        style(ax)
        mean, lo, hi, pk_t, pk_v, pk_ci = peak_stats(traces[key])
        summary.append((label, color, pk_t, pk_ci))

        ax.axhline(0, color=GRIDC, lw=1, zorder=1)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.fill_between(GRID, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(GRID, mean, color=color, lw=2, zorder=3, solid_capstyle='round')
        ax.plot([pk_t], [pk_v], 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
        ax.set_ylabel(f'{label}\n({unit})', fontsize=9.5, color=INK, linespacing=1.5)
        ax.annotate(
            f'peak {pk_t:+.2f} s  [{pk_ci[0]:+.2f}, {pk_ci[1]:+.2f}]',
            xy=(0.985, 0.92),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=8.5,
            color=INK2,
        )

    axes[0].set_xlim(GRID[0], GRID[-1])
    axes[-1].set_xlabel('Time since commanded release onset (s)', fontsize=9.5, color=INK2)

    lead_pos, k_pos, _, p_pos = sign_test(traces['tac'], traces['relw'])
    lead_vel, k_vel, _, p_vel = sign_test(traces['tac'], traces['wvel'])

    fig.suptitle(
        'Controlled slip: what each channel reports after the policy commands a release',
        fontsize=13,
        color=INK,
        x=0.012,
        ha='left',
        y=0.988,
    )
    fig.text(
        0.012,
        0.957,
        f'{counts[0]} of {counts[1]} deployed rollouts, {n} commanded-release events. '
        'Mean per channel with a bootstrap 95% band.\n'
        f'Tactile leads the width RESIDUAL in {k_pos}/{n} events (p = {p_pos:.1e}),\n'
        f'but only ties gripper SPEED ({k_vel}/{n}, p = {p_vel:.2f}) -- the lead is over '
        'position readout,\nnot over proprioception that differentiates.',
        fontsize=9.3,
        color=INK2,
        va='top',
        linespacing=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    _draw_summary(summary, n, out.with_name(out.stem + '_peaks'))
    return {
        'n_events': n,
        'rollouts': counts,
        'pos': (lead_pos, k_pos, p_pos),
        'vel': (lead_vel, k_vel, p_vel),
        'peaks': {lbl: (t, ci) for lbl, _c, t, ci in summary},
    }


def _draw_summary(summary, n: int, out: pathlib.Path) -> None:
    """Peak latency per channel with its CI -- the 'which reports first' headline."""
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis='x', color=GRIDC, lw=0.8, zorder=0)
    ax.grid(axis='y', visible=False)

    for y, (label, color, pk_t, pk_ci) in enumerate(summary):
        ax.plot(pk_ci, [y, y], color=color, lw=3, solid_capstyle='round', alpha=0.45, zorder=2)
        ax.plot([pk_t], [y], 'o', ms=9, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(
            f'{pk_t:+.2f} s',
            xy=(pk_t, y),
            xytext=(0, 11),
            textcoords='offset points',
            ha='right' if pk_t > 1.4 else 'center',
            fontsize=8.5,
            color=INK2,
        )
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([r[0] for r in summary], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(summary) - 0.4, -0.6)
    ax.set_xlim(GRID[0], GRID[-1])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Peak latency after commanded release (s)', fontsize=9.5, color=INK2)
    fig.suptitle(
        f'Peak latency per channel (bootstrap 95% CI), n = {n} events',
        fontsize=12,
        color=INK,
        x=0.012,
        ha='left',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    """Build the reviewer-response timing figures from the extracted rollout archives."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', type=pathlib.Path, required=True, help='dir of *_full.npz')
    ap.add_argument('--out', type=pathlib.Path, required=True, help='output path stem')
    args = ap.parse_args()

    events, counts = load_events(args.store)
    print(f'rollouts analysed: {counts[0]} of {counts[1]}')
    print(f'commanded-release events: {len(events)}')
    stats = draw(events, counts, args.out)
    for label, (t, ci) in stats['peaks'].items():
        print(f'  {label:>28}: peak {t:+.2f} s  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]')
    print(f'  tactile vs width residual: {stats["pos"][1]}/{stats["n_events"]}, p={stats["pos"][2]:.2e}')
    print(f'  tactile vs gripper speed : {stats["vel"][1]}/{stats["n_events"]}, p={stats["vel"][2]:.2f}')


if __name__ == '__main__':
    main()
