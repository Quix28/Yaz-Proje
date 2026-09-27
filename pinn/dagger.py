"""
Step 5b: DAgger. Fixes compounding-error drift from pure imitation.

Each round: roll out the current PINN closed-loop (against the same torch
plant used everywhere), collect the states it actually visits, relabel them
with the MPC teacher, append to the dataset, and retrain (warm-started from
the previous round). 2-3 rounds.

Reuses losses.rk4_step (identical plant to sim_loop._rk4_step_torch and the
physics-loss rollout) so the DAgger plant, training rollout, and evaluation
all agree.
"""
import os

import numpy as np
import torch

from pinn import config as C
from pinn import dataset as ds
from pinn import param_utils as pu
from pinn import losses as L
from pinn import train as T
from pinn.model import load_checkpoint
from pinn.actuator import voltage_to_force


@torch.no_grad()
def _rollout_pinn(model, params, mlparams, x0, steps, dt):
    """Closed-loop rollout under the PINN. Returns list of visited states."""
    batched = pu.batched_torch_params(np.asarray(mlparams).reshape(1, 4), dtype=torch.float64)
    ml_t = torch.tensor(mlparams, dtype=torch.float64).unsqueeze(0)
    x = torch.tensor(x0, dtype=torch.float64)
    visited = []
    for _ in range(steps):
        V = model(x.unsqueeze(0), ml_t).to(torch.float64).squeeze(0)
        F = voltage_to_force(V, x[3])
        x = L.rk4_step(x.unsqueeze(0), F.unsqueeze(0), batched, dt).squeeze(0)
        if not torch.isfinite(x).all() or x.abs().max() > 10:
            break                       # diverged; stop this rollout
        visited.append(x.numpy().copy())
    return visited


def _relabel(job):
    """Worker: label one config's visited states with the seed-set teacher."""
    ml, states = job
    ctrl = ds.make_teacher(ml)
    return [(x, u0) for x in states if (u0 := ds.label(ctrl, x)) is not None]


def run_round(round_idx, init_ckpt, seed=None, verbose=True,
              dataset_path=None, out_dir=None, ckpt_dir=None, use_wandb=False):
    """
    One DAgger round. Returns (dataset_path, ckpt_path, n_added) -- THREE values.

    This docstring previously claimed two. A driver written against it did
    `ds_path, ck = run_round(...)` and died with "too many values to unpack"
    AFTER round 1 had already saved, which silently cost rounds 2 and 3.
    Prefer run() below over hand-written drivers.

    out_dir/ckpt_dir default to C.DATA_DIR/C.CKPT_DIR (the real project
    dirs) -- override them (e.g. to a tempdir) for smoke/e2e testing so
    a test run can't leak round-N artifacts into the real project state.
    """
    seed = (C.SEED + round_idx) if seed is None else seed
    out_dir = out_dir or C.DATA_DIR
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    rng = np.random.default_rng(seed)
    model, _ = load_checkpoint(init_ckpt)
    model.eval()

    # roll out under a mix of configs (reuse the dataset sampler)
    configs = pu.sample_configs(C.DAGGER_CONFIGS, rng=rng)
    new_states, new_ml, new_u, new_cid = [], [], [], []
    base = ds.load_dataset(dataset_path)
    next_cid = int(base["config_id"].max()) + 1
    n_added = 0

    jobs = []
    for ml in configs:
        params = pu.full_params_from_ml(*ml)
        collected = []
        ics = ds._sample_states(C.DAGGER_ICS, rng)  # same off-center/push mix as the seed set
        for x0 in ics:
            visited = _rollout_pinn(model, params, ml, x0, C.DAGGER_STEPS, C.DT)
            collected.extend(visited[::C.DAGGER_SUBSAMPLE])   # subsample
        jobs.append((ml, collected))

    # relabel in parallel across configs: cold solves are ~1 s each at Np=40,
    # so a serial round would take about an hour
    import multiprocessing as mp
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for ci, ((ml, collected), labeled) in enumerate(zip(jobs, pool.imap(_relabel, jobs))):
            for x, u0 in labeled:
                new_states.append(x)
                new_ml.append(np.asarray(ml, dtype=np.float64))
                new_u.append(u0)
                new_cid.append(next_cid + ci)
                n_added += 1
            if verbose:
                print(f"[dagger round {round_idx}] config {ci + 1}/{len(configs)} done "
                      f"({len(collected)} candidates, {n_added} kept so far)", flush=True)

    # assemble round dataset = seed + all relabeled DAgger points
    if n_added > 0:
        data = dict(
            states=np.concatenate([base["states"], np.asarray(new_states)], 0),
            mlparams=np.concatenate([base["mlparams"], np.asarray(new_ml)], 0),
            u=np.concatenate([base["u"], np.asarray(new_u)], 0),
            config_id=np.concatenate([base["config_id"], np.asarray(new_cid, dtype=np.int32)], 0),
        )
    else:
        data = base

    ds_path = os.path.join(out_dir, f"dataset_round{round_idx}.npz")
    np.savez(ds_path, **data)
    # keep seed norm stats (comparable splits across rounds) -- do NOT recompute

    if verbose:
        print(f"[dagger round {round_idx}] added {n_added} relabeled points "
              f"-> {ds_path}")

    ckpt_path = os.path.join(ckpt_dir, f"round{round_idx}_best.pt")
    T.train(dataset_path=ds_path, out_ckpt=ckpt_path, init_ckpt=init_ckpt,
            seed=seed, verbose=verbose, use_wandb=use_wandb,
            wandb_run_name=f"dagger-round{round_idx}", wandb_group="dagger")
    return ds_path, ckpt_path, n_added


def run(rounds=None, seed_ckpt=None, verbose=True, use_wandb=False,
        start_round=1, dataset_path=None, out_dir=None, ckpt_dir=None):
    """Run DAgger rounds starting from a seed-trained checkpoint.

    Each round's assembled dataset (seed + all relabeled points so far) is
    threaded into the next round's `dataset_path` -- otherwise every round
    would silently reload the raw seed dataset and DAgger would never
    actually accumulate visited-state data across rounds.

    start_round/dataset_path exist to RESUME an interrupted sequence without
    redoing completed rounds. To continue after round 1, pass that round's
    checkpoint as seed_ckpt and its dataset as dataset_path -- omitting the
    dataset silently falls back to the raw seed set and throws away every
    relabeled point round 1 produced.
    """
    rounds = C.DAGGER_ROUNDS if rounds is None else rounds
    ckpt = seed_ckpt or os.path.join(C.CKPT_DIR, "round0_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"seed checkpoint not found: {ckpt} -- run `python -m pinn.train` "
            f"(Step 5a) first to produce it before starting DAgger."
        )
    if start_round > 1 and dataset_path is None:
        raise ValueError(
            f"resuming at round {start_round} without --dataset would discard every "
            f"point earlier rounds relabeled; pass the previous round's dataset .npz"
        )
    ds_path = dataset_path
    for k in range(start_round, rounds + 1):
        ds_path, ckpt, _ = run_round(k, ckpt, dataset_path=ds_path, verbose=verbose,
                                     use_wandb=use_wandb, out_dir=out_dir,
                                     ckpt_dir=ckpt_dir)
    return ckpt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DAgger rounds, resumable.")
    ap.add_argument("--rounds", type=int, default=None, help=f"last round (default {C.DAGGER_ROUNDS})")
    ap.add_argument("--from-round", type=int, default=1, help="first round to run; >1 resumes")
    ap.add_argument("--ckpt", default=None, help="checkpoint to warm-start from")
    ap.add_argument("--dataset", default=None,
                    help="dataset .npz to build on; REQUIRED when --from-round > 1")
    ap.add_argument("--out-dir", default=None, help="round datasets (default pinn/data)")
    ap.add_argument("--ckpt-dir", default=None, help="round checkpoints (default pinn/checkpoints)")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    for d in (a.out_dir, a.ckpt_dir):
        if d:
            os.makedirs(d, exist_ok=True)
    run(rounds=a.rounds, seed_ckpt=a.ckpt, use_wandb=a.wandb,
        start_round=a.from_round, dataset_path=a.dataset,
        out_dir=a.out_dir, ckpt_dir=a.ckpt_dir)
