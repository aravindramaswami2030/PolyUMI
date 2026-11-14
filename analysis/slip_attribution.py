"""
Per-modality attribution for a slip-control diffusion policy.

Answers the reviewer's question -- what does each modality contribute *when all are present* --
which is different from what the trained ablations (slip_vt, slip_va) answer. Those retrain
without a modality; this holds the full model fixed and interferes with one input at a time.

Two independent methods, because agreement between them is far stronger evidence than either:

* OCCLUSION: replace one modality with a baseline and measure how far the predicted action moves.
  Several baselines are reported, because each is wrong in its own way -- zeros are off-manifold
  (the model never saw silence), the dataset mean is on-manifold but destroys all structure, and
  a time-shuffle keeps each modality's marginal statistics while destroying its synchrony with
  the others, which isolates *timing* information specifically.

* INTEGRATED GRADIENTS: fade each input from a baseline to its real value, averaging the gradient
  of the action along that path. Gives a signed, roughly additive attribution per input element,
  summable within a modality and resolvable per timestep.

The diffusion sampler is stochastic, so every condition runs under a fixed noise seed; otherwise
the occlusion effect is buried in resampling spread. That spread is measured here and reported as
the noise floor any effect must clear.
"""

import argparse
import sys

import dill
import hydra
import numpy as np
import torch


def fixed_seed_action(policy, obs, seed: int, n_seeds: int = 1) -> torch.Tensor:
    """
    Predict the EXPECTED action chunk, averaged over `n_seeds` sampler draws.

    A single draw is useless here: this policy's per-call sampling spread (~0.4 mm) is larger than
    its own mean output (~0.34 mm), so one sample is mostly noise. Averaging shrinks that noise as
    1/sqrt(n_seeds) while leaving the conditioning effect -- the thing being measured -- untouched.
    Every condition uses the same seed set, so the comparison stays paired.
    """
    out = None
    with torch.no_grad():
        for i in range(n_seeds):
            torch.manual_seed(seed + 1000 * i)
            a = policy.predict_action(obs)['action_pred']
            out = a if out is None else out + a
    return out / n_seeds


def corrupt(obs: dict, key: str, mode: str, stats: dict, generator: torch.Generator) -> dict:
    """Return a copy of `obs` with one modality replaced, leaving every other key untouched."""
    out = {k: v.clone() for k, v in obs.items()}
    x = out[key]
    if mode == 'zeros':
        out[key] = torch.zeros_like(x)
    elif mode == 'mean':
        out[key] = torch.full_like(x, float(stats[key]))
    elif mode == 'shuffle':
        # Swap in this modality from an unrelated sample drawn far away in the dataset. Rolling
        # within the batch is NOT enough: consecutive samples are overlapping sequence windows, so
        # the neighbour is a near-identical copy and the "corruption" changes nothing.
        donor = stats.get(key + '_donor')
        if donor is not None:
            out[key] = donor[: x.shape[0]].to(x.device)
    elif mode == 'noise':
        out[key] = x + torch.randn(x.shape, generator=generator, device=x.device) * float(stats[key + '_std'])
    else:
        raise ValueError(f'unknown corruption {mode}')
    return out


def integrated_gradients(policy, obs: dict, key: str, seed: int, steps: int) -> float:
    """
    Total |attribution| of one modality, by fading it in from zero and integrating the gradient.

    The other modalities stay at their real values throughout, so this measures the contribution
    of `key` in the context the model actually sees, not in isolation.
    """
    baseline = torch.zeros_like(obs[key])
    real = obs[key]
    total = torch.zeros_like(real)
    for i in range(steps):
        alpha = (i + 0.5) / steps
        probe = {k: v.clone() for k, v in obs.items()}
        probe[key] = (baseline + alpha * (real - baseline)).requires_grad_(True)
        torch.manual_seed(seed)
        action = policy.predict_action(probe)['action_pred']
        grad = torch.autograd.grad(action.abs().sum(), probe[key], retain_graph=False, allow_unused=True)[0]
        if grad is not None:
            total = total + grad.detach()
    attribution = (real - baseline) * total / steps
    return attribution.abs().sum(dim=tuple(range(1, attribution.ndim))).mean().item()


def main() -> int:
    """Run occlusion and IG over held-out samples and print a per-modality table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', required=True, help="path, or 'auto' to use the checkpoint's own")
    ap.add_argument('--samples', type=int, default=128, help='held-out samples to evaluate')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ig-steps', type=int, default=0, help='integrated-gradient steps; 0 skips IG')
    ap.add_argument('--seed-avg', type=int, default=8, help='sampler draws averaged per condition')
    ap.add_argument('--device', default='cuda')
    ap.add_argument(
        '--select',
        choices=('first', 'active'),
        default='active',
        help='active: rank held-out samples by |ground-truth action| and take the largest, so '
        'attribution is measured where the policy actually moves the gripper rather than on '
        'quiescent holding, where every effect washes out below the sampling floor.',
    )
    args = ap.parse_args()

    payload = torch.load(open(args.ckpt, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    ws = hydra.utils.get_class(cfg._target_)(cfg)
    ws.load_payload(payload)
    policy = ws.ema_model.to(args.device).eval()

    if args.dataset != 'auto':
        cfg.task.dataset.dataset_path = args.dataset
    print(f'dataset   : {cfg.task.dataset.dataset_path}')
    val = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()

    if args.select == 'active':
        scan = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
        mags = torch.cat([b['action'].abs().mean(dim=(1, 2)) for b in scan])
        order = torch.argsort(mags, descending=True)
        chosen = order[: args.samples].tolist()
        donors = order[-args.samples :].tolist()  # quietest samples, used as shuffle donors
        print(
            f'selection : |action| p50={mags.median():.5f} max={mags.max():.5f}; '
            f'chosen mean={mags[order[: args.samples]].mean():.5f}'
        )
        val = torch.utils.data.Subset(val, chosen + donors)
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0)
    modalities = list(cfg.task.shape_meta.obs.keys())
    print(
        f'model     : {cfg.policy._target_.split(".")[-1]}  ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M)'
    )
    print(f'modalities: {modalities}')
    print(f'val       : {len(val)} samples, evaluating {args.samples}\n')

    modes = ['zeros', 'mean', 'shuffle']
    # Pairs, to separate redundancy from irrelevance. Occluding one modality only measures its
    # MARGINAL contribution: if two streams carry the same information, each alone scores near
    # zero because the other compensates, and only removing both reveals it. Super-additive
    # Delta(pair) >> Delta(a) + Delta(b) is the signature of redundancy; additive is independence.
    pairs = [(a, b) for i, a in enumerate(modalities) for b in modalities[i + 1 :]]
    pair_deltas = {p: {'zeros': [], 'shuffle': []} for p in pairs}
    deltas = {m: {c: [] for c in modes} for m in modalities}
    ig = {m: [] for m in modalities}
    floor, action_mag, all_swap, all_zero = [], [], [], []
    gen = torch.Generator(device=args.device)
    seen = 0
    donor_bank = {}

    for batch in loader:
        if seen >= args.samples:
            break
        obs = {k: v.to(args.device) for k, v in batch['obs'].items()}
        stats = {}
        for k, v in obs.items():
            stats[k] = v.mean().item()
            stats[k + '_std'] = v.std().item()
            if donor_bank.get(k) is not None:
                stats[k + '_donor'] = donor_bank[k]

        base = fixed_seed_action(policy, obs, args.seed, args.seed_avg)
        # Floor: a second, independent seed set on the SAME input. Whatever separates these two is
        # residual sampler noise surviving the averaging, and a real effect must exceed it.
        alt = fixed_seed_action(policy, obs, args.seed + 500_000, args.seed_avg)
        floor.append((base - alt).abs().mean().item())
        action_mag.append(base.abs().mean().item())

        if not donor_bank:  # first batch doubles as the donor pool for later batches
            donor_bank = {k: v.clone() for k, v in obs.items()}
        for key in modalities:
            for mode in modes:
                gen.manual_seed(args.seed)
                changed = fixed_seed_action(policy, corrupt(obs, key, mode, stats, gen), args.seed, args.seed_avg)
                deltas[key][mode].append((base - changed).abs().mean().item())
            if args.ig_steps:
                ig[key].append(integrated_gradients(policy, obs, key, args.seed, args.ig_steps))
        # Whole-observation perturbations: the upper bound on how much observations matter.
        for mode, bucket in (('shuffle', all_swap), ('zeros', all_zero)):
            wrecked = obs
            for key in modalities:
                gen.manual_seed(args.seed)
                wrecked = corrupt(wrecked, key, mode, stats, gen)
            bucket.append((base - fixed_seed_action(policy, wrecked, args.seed, args.seed_avg)).abs().mean().item())

        for a, b in pairs:
            for jmode in ('zeros', 'shuffle'):
                gen.manual_seed(args.seed)
                both = corrupt(corrupt(obs, a, jmode, stats, gen), b, jmode, stats, gen)
                changed = fixed_seed_action(policy, both, args.seed, args.seed_avg)
                pair_deltas[(a, b)][jmode].append((base - changed).abs().mean().item())
        seen += obs[modalities[0]].shape[0]

    print(f'sampler draws averaged   : {args.seed_avg}')
    print(f'mean |action|            : {np.mean(action_mag):.5f}')
    print(f'sampling floor (seed sw) : {np.mean(floor):.5f}   <- effects below this are noise\n')
    head = f'{"modality":>22}' + ''.join(f'{m:>12}' for m in modes) + f'{"xfloor":>9}'
    print(head + ('' if not args.ig_steps else f'{"IG":>12}'))
    print('-' * len(head))
    for key in modalities:
        row = f'{key:>22}'
        for mode in modes:
            row += f'{np.mean(deltas[key][mode]):>12.5f}'
        best = max(np.mean(deltas[key][m]) for m in modes)
        row += f'{best / max(np.mean(floor), 1e-12):>9.1f}'
        if args.ig_steps:
            row += f'{np.mean(ig[key]):>12.4f}'
        print(row)

    print('\n=== sanity: is the policy conditioned on its observations at all? ===')
    print(f'  swap EVERY modality for another sample  : {np.mean(all_swap):.5f}')
    print(f'  zero  EVERY modality                    : {np.mean(all_zero):.5f}')
    print(f'  floor                                   : {np.mean(floor):.5f}')
    print(f'  mean |predicted action|                 : {np.mean(action_mag):.5f}')
    print('  if the swap is near the floor, the policy is effectively open-loop and no')
    print('  per-modality attribution can be meaningful.')

    print('\n=== joint occlusion (both zeroed) ===')
    print(f'{"pair":>40}{"joint0":>9}{"sum0":>9}{"r0":>6}{"jointS":>9}{"sumS":>9}{"rS":>6}')
    print('-' * 88)
    for a, b in pairs:
        row = f'{a + " + " + b:>40}'
        for jmode, tag in (('zeros', 'zeros'), ('shuffle', 'shuffle')):
            joint = np.mean(pair_deltas[(a, b)][jmode])
            singly = np.mean(deltas[a][tag]) + np.mean(deltas[b][tag])
            row += f'{joint:>9.5f}{singly:>9.5f}{joint / max(singly, 1e-12):>6.2f}'
        print(row)
    print('\nratio >> 1 means redundancy (each masks the other); ~1 means independent contributions.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
