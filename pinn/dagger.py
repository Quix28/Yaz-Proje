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
from mpc import MOTOR_FORCE_MAX


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


def run_round(round_idx, init_ckpt, seed=None, verbose=True,
              dataset_path=None, out_dir=None, ckpt_dir=None):
    """
    One DAgger round. Returns (dataset_path, ckpt_path).

    out_dir/ckpt_dir default to C.DATA_DIR/C.CKPT_DIR (the real project
    dirs) -- override them (e.g. to a tempdir) for smoke/e2e testing so
    a test run can't leak round-N artifacts into the real project state.
    """
    seed = (C.SEED + round_idx) if seed is None else seed
    out_dir = out_dir or C.DATA_DIR
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    rng = np.random.default_rng(seed)
    from mpc import MPCController

    model, _ = load_checkpoint(init_ckpt)
    model.eval()

    # roll out under a mix of configs (reuse the dataset sampler)
    configs = pu.sample_configs(C.DAGGER_CONFIGS, rng=rng)
    new_states, new_ml, new_u, new_cid = [], [], [], []
    base = ds.load_dataset(dataset_path)
    next_cid = int(base["config_id"].max()) + 1
    n_added = 0

    for ci, ml in enumerate(configs):
        params = pu.full_params_from_ml(*ml)
        ctrl = MPCController(params, Np=C.MPC_NP, dt=C.DT, s_max=C.S_MAX)

        collected = []
        ics = ds._sample_states(C.DAGGER_ICS, rng)  # same off-center/push mix as the seed set
        for x0 in ics:
            visited = _rollout_pinn(model, params, ml, x0, C.DAGGER_STEPS, C.DT)
            collected.extend(visited[::C.DAGGER_SUBSAMPLE])   # subsample

        # relabel visited states with the warm-started teacher
        for x in collected:
            try:
                u0, _, _ = ctrl.solve(x)
            except RuntimeError:
                ctrl._X_prev = np.zeros_like(ctrl._X_prev)
                ctrl._U_prev = np.zeros_like(ctrl._U_prev)
                continue
            if not np.isfinite(u0) or abs(u0) > C.MAX_LABEL_FACTOR * MOTOR_FORCE_MAX:
                continue
            new_states.append(x)
            new_ml.append(np.asarray(ml, dtype=np.float64))
            new_u.append(u0)
            new_cid.append(next_cid + ci)
            n_added += 1

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
            seed=seed, verbose=verbose)
    return ds_path, ckpt_path, n_added


def run(rounds=None, seed_ckpt=None, verbose=True):
    """Run all DAgger rounds starting from the seed-trained checkpoint.

    Each round's assembled dataset (seed + all relabeled points so far) is
    threaded into the next round's `dataset_path` -- otherwise every round
    would silently reload the raw seed dataset and DAgger would never
    actually accumulate visited-state data across rounds.
    """
    rounds = C.DAGGER_ROUNDS if rounds is None else rounds
    ckpt = seed_ckpt or os.path.join(C.CKPT_DIR, "round0_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"seed checkpoint not found: {ckpt} -- run `python -m pinn.train` "
            f"(Step 5a) first to produce it before starting DAgger."
        )
    ds_path = None
    for k in range(1, rounds + 1):
        ds_path, ckpt, _ = run_round(k, ckpt, dataset_path=ds_path, verbose=verbose)
    return ckpt


if __name__ == "__main__":
    run()
