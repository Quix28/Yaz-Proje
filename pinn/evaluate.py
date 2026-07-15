"""
Step 6: closed-loop evaluation.

  * rollout()/rollout_metrics(): settling time, peak deviation, control
    effort, success rate for one policy under one config+IC.
  * run_ablation(): data-only vs +physics vs +physics+barrier vs full --
    proves the physics/barrier/EL terms matter (or doesn't; report either way).
  * run_generalization(): full PINN vs the plain-imitation-NN and LQR
    baselines, on held-out INTERPOLATED (inside the training (m,l) box) and
    EXTRAPOLATED (inside the wider collocation box, outside the training
    box) configs.

All rollouts share pinn.losses.rk4_step -- the same plant used by training
and DAgger -- so evaluation is dynamically consistent with what the model
was trained/rolled-out against.
"""
import argparse
import os

import numpy as np
import torch

from pinn import config as C
from pinn import param_utils as pu
from pinn import dataset as ds
from pinn import losses as L
from pinn.model import load_checkpoint
from pinn.actuator import voltage_to_force
from pinn.baselines import lqr_gain, lqr_policy, train_plain_imitation
from pinn.train import train
from mpc import MOTOR_FORCE_MAX

# absolute per-state tolerance for "settled": s[m], th1/th2[rad], sdot, th1dot, th2dot
SETTLE_TOL = np.array([0.02, 0.05, 0.05, 0.05, 0.2, 0.2])
SETTLE_WINDOW_FRAC = 0.2   # must stay inside SETTLE_TOL for the trailing 20% of the rollout


def rollout(policy_fn, mlparams, x0, steps, dt=None):
    """
    Closed-loop rollout under a force-output policy_fn(state_ndarray)->force.
    Returns dict(states (T,6) ndarray, forces (T-1,) ndarray, diverged:bool).
    """
    dt = dt or C.DT
    batched = pu.batched_torch_params(np.asarray(mlparams).reshape(1, 4))
    x = torch.tensor(x0, dtype=torch.float64)
    states, forces = [x.numpy().copy()], []
    diverged = False
    for _ in range(steps):
        F = float(np.clip(policy_fn(x.numpy()), -MOTOR_FORCE_MAX, MOTOR_FORCE_MAX))
        forces.append(F)
        x = L.rk4_step(x.unsqueeze(0), torch.tensor([F], dtype=torch.float64),
                       batched, dt).squeeze(0)
        if not torch.isfinite(x).all() or x.abs().max() > 10 or abs(float(x[0])) > 1.5 * C.S_MAX:
            diverged = True
            break
        states.append(x.numpy().copy())
    return dict(states=np.array(states), forces=np.array(forces), diverged=diverged)


def rollout_metrics(traj, dt=None):
    """settling time [s], peak |s|/|th1|/|th2|, control effort (sum F^2 dt), success."""
    dt = dt or C.DT
    states, forces = traj["states"], traj["forces"]
    T = len(states)
    within_tol = np.all(np.abs(states) <= SETTLE_TOL, axis=1)
    win = max(1, int(SETTLE_WINDOW_FRAC * T))
    settle_idx = next((t for t in range(T - win) if within_tol[t:].all()), None)
    settled = settle_idx is not None
    success = settled and not traj["diverged"] and float(np.abs(states[:, 0]).max()) <= C.S_MAX
    return dict(
        success=bool(success),
        diverged=bool(traj["diverged"]),
        settling_time=float(settle_idx * dt) if settled else float("nan"),
        peak_s=float(np.abs(states[:, 0]).max()),
        peak_th1=float(np.abs(states[:, 1]).max()),
        peak_th2=float(np.abs(states[:, 2]).max()),
        control_effort=float(np.sum(forces ** 2) * dt),
        steps_survived=T,
    )


def evaluate_policy(make_policy, configs, n_ics, steps, seed=0, dt=None):
    """
    make_policy(mlparams) -> policy_fn(state_ndarray)->force.
    configs: (n_configs, 4) ndarray of [m1,m2,l1,l2].
    Returns a flat list of per-rollout metric dicts (config_id, ml attached).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ci, ml in enumerate(configs):
        policy = make_policy(np.asarray(ml))
        for x0 in ds._sample_states(n_ics, rng):
            m = rollout_metrics(rollout(policy, ml, x0, steps, dt), dt)
            m["config_id"], m["ml"] = ci, tuple(float(v) for v in ml)
            rows.append(m)
    return rows


def summarize(rows):
    settle = [r["settling_time"] for r in rows if r["success"]]
    return dict(
        n=len(rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        settling_time_mean=float(np.mean(settle)) if settle else float("nan"),
        control_effort_mean=float(np.mean([r["control_effort"] for r in rows])),
        peak_s_mean=float(np.mean([r["peak_s"] for r in rows])),
        peak_th1_mean=float(np.mean([r["peak_th1"] for r in rows])),
        peak_th2_mean=float(np.mean([r["peak_th2"] for r in rows])),
    )


def make_pinn_policy(ckpt_path):
    model, _ = load_checkpoint(ckpt_path)
    model.eval()

    def make_policy(mlparams):
        ml_t = torch.tensor(mlparams, dtype=torch.float64).unsqueeze(0)

        @torch.no_grad()
        def policy(state):
            x = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
            V = model(x, ml_t).to(torch.float64).squeeze(0)
            return float(voltage_to_force(V, x[0, 3]))
        return policy
    return make_policy


def make_lqr_policy(**lqr_kwargs):
    def make_policy(mlparams):
        return lqr_policy(lqr_gain(mlparams, **lqr_kwargs))
    return make_policy


def sample_interp_configs(n, rng):
    """Held-out configs inside the training (m,l) box -- same distribution, unseen combos."""
    return pu.sample_configs(n, rng=rng)


def sample_extrap_configs(n, rng):
    """Configs inside the wider collocation box but OUTSIDE the training box -- never seen."""
    low, high = np.array(C.PARAM_LOW), np.array(C.PARAM_HIGH)
    clow, chigh = np.array(C.COLLOC_PARAM_LOW), np.array(C.COLLOC_PARAM_HIGH)
    out = []
    while len(out) < n:
        cand = rng.uniform(clow, chigh)
        if np.any(cand < low) or np.any(cand > high):
            out.append(cand)
    return np.array(out)


ABLATIONS = {
    "data_only": {"w_phys": 0.0, "w_bar": 0.0, "w_el": 0.0},
    "plus_physics": {"w_bar": 0.0, "w_el": 0.0},
    "plus_physics_barrier": {"w_el": 0.0},
    "full": None,
}


def run_ablation(full_ckpt, dataset_path=None, ckpt_dir=None, configs=None,
                 n_configs=10, n_ics=8, steps=150, seed=0, verbose=True):
    """
    Trains the three missing ablation variants (skips ones already checkpointed)
    then evaluates all four on the same held-out (interpolated) config set.
    Returns dict(variant -> summary dict).
    """
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    rng = np.random.default_rng(seed)
    configs = sample_interp_configs(n_configs, rng) if configs is None else configs

    results = {}
    for name, overrides in ABLATIONS.items():
        if name == "full":
            ckpt = full_ckpt
        else:
            ckpt = os.path.join(ckpt_dir, f"ablation_{name}.pt")
            if not os.path.exists(ckpt):
                if verbose:
                    print(f"[ablation] training {name}...")
                train(dataset_path=dataset_path, out_ckpt=ckpt, seed=seed,
                     verbose=verbose, weight_overrides=overrides)
        rows = evaluate_policy(make_pinn_policy(ckpt), configs, n_ics, steps, seed=seed)
        results[name] = summarize(rows)
        if verbose:
            s = results[name]
            print(f"[ablation] {name:22s} success={s['success_rate']:.2f} "
                 f"settle={s['settling_time_mean']:.2f}s effort={s['control_effort_mean']:.1f} "
                 f"peak_th1={s['peak_th1_mean']:.3f}")
    return results


def run_generalization(full_ckpt, plain_ckpt=None, dataset_path=None, ckpt_dir=None,
                       n_configs=10, n_ics=8, steps=150, seed=0, verbose=True):
    """
    full PINN vs plain-imitation-NN vs LQR, each run on both an interpolation
    and an extrapolation held-out config split. Returns dict(split -> dict(controller -> summary)).
    """
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    plain_ckpt = plain_ckpt or os.path.join(ckpt_dir, "ablation_data_only.pt")
    if not os.path.exists(plain_ckpt):
        if verbose:
            print("[generalization] training plain-imitation-NN baseline...")
        train_plain_imitation(dataset_path=dataset_path, out_ckpt=plain_ckpt,
                             seed=seed, verbose=verbose)

    rng = np.random.default_rng(seed)
    splits = {
        "interpolation": sample_interp_configs(n_configs, rng),
        "extrapolation": sample_extrap_configs(n_configs, rng),
    }
    controllers = {
        "full_pinn": make_pinn_policy(full_ckpt),
        "plain_imitation_nn": make_pinn_policy(plain_ckpt),
        "lqr": make_lqr_policy(),
    }

    results = {}
    for split_name, configs in splits.items():
        results[split_name] = {}
        for ctrl_name, make_policy in controllers.items():
            rows = evaluate_policy(make_policy, configs, n_ics, steps, seed=seed)
            results[split_name][ctrl_name] = summarize(rows)
            if verbose:
                s = results[split_name][ctrl_name]
                print(f"[generalization] {split_name:14s} {ctrl_name:20s} "
                     f"success={s['success_rate']:.2f} settle={s['settling_time_mean']:.2f}s "
                     f"effort={s['control_effort_mean']:.1f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ckpt", default=os.path.join(C.CKPT_DIR, "round0_best.pt"))
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--generalization", action="store_true")
    ap.add_argument("--n-configs", type=int, default=10)
    ap.add_argument("--n-ics", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    if not args.ablation and not args.generalization:
        args.ablation = args.generalization = True

    if args.ablation:
        run_ablation(args.full_ckpt, n_configs=args.n_configs, n_ics=args.n_ics, steps=args.steps)
    if args.generalization:
        run_generalization(args.full_ckpt, n_configs=args.n_configs, n_ics=args.n_ics, steps=args.steps)
