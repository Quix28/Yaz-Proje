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
import json
import subprocess
import time

import numpy as np
import torch

from pinn import config as C
from pinn import param_utils as pu
from pinn import dataset as ds
from pinn import losses as L
from pinn.model import load_checkpoint
from pinn.actuator import voltage_to_force
from pinn.baselines import lqr_gain, lqr_policy, lqr_cost_to_go
from pinn.train import train
from mpc import MOTOR_FORCE_MAX

# absolute per-state tolerance for "settled": s[m], th1/th2[rad], sdot, th1dot, th2dot
SETTLE_TOL = np.array([0.02, 0.05, 0.05, 0.05, 0.2, 0.2])
SETTLE_WINDOW_FRAC = 0.2   # must stay inside SETTLE_TOL for the trailing 20% of the rollout

# Slack on the peak-|s| success test. The MPC's cart-position bound is a HARD
# constraint, so the optimizer is entitled to ride it exactly to S_MAX, and IPOPT
# satisfies constraints only to its own tolerance (~1e-8). Testing peak_s <= S_MAX
# with no slack therefore failed rollouts that had settled cleanly and never
# diverged, purely because the cart overshot by ~1e-8 m -- 10 nanometres. That
# suppressed roughly 20 percentage points of measured success across the horizon
# sweep. 1e-6 is three orders above solver tolerance and six below any excursion
# that means anything physically.
S_MAX_SLACK = 1e-6


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
    # range(T - win + 1): the +1 matters -- without it a trajectory that settles
    # exactly at the start of the trailing window is scored as never settling.
    settle_idx = next((t for t in range(T - win + 1) if within_tol[t:].all()), None)
    settled = settle_idx is not None
    success = (settled and not traj["diverged"]
               and float(np.abs(states[:, 0]).max()) <= C.S_MAX + S_MAX_SLACK)
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


def sample_center_states(n, rng):
    """Small perturbations from upright only -- no off-centre/push regimes.
    Lets a caller separate 'can it balance' from 'can it also recentre the
    cart from 0.16 m out', which the mixture sampler conflates."""
    return rng.uniform(-C.STATE_PERT, C.STATE_PERT, size=(n, 6))


def evaluate_policy(make_policy, configs, n_ics, steps, seed=0, dt=None,
                    ic_sampler=None):
    """
    make_policy(mlparams) -> policy_fn(state_ndarray)->force.
    configs: (n_configs, 4) ndarray of [m1,m2,l1,l2].
    ic_sampler(n, rng) -> (n,6) initial states; defaults to the dataset's
    mixture sampler. Pass sample_center_states for centre-only ICs.
    Returns a flat list of per-rollout metric dicts (config_id, ml attached).
    """
    rng = np.random.default_rng(seed)
    ic_sampler = ic_sampler or ds._sample_states
    rows = []
    for ci, ml in enumerate(configs):
        policy = make_policy(np.asarray(ml))
        for x0 in ic_sampler(n_ics, rng):
            m = rollout_metrics(rollout(policy, ml, x0, steps, dt), dt)
            m["config_id"], m["ml"] = ci, tuple(float(v) for v in ml)
            rows.append(m)
    return rows


def summarize(rows, steps=None):
    """Strict success plus GRADED metrics. Binary success saturates at 0 when
    the plant is near-infeasible, which hides real differences between
    controllers -- survival and peak angle still discriminate there."""
    settle = [r["settling_time"] for r in rows if r["success"]]
    surv = [r["steps_survived"] for r in rows]
    out = dict(
        n=len(rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        diverged_rate=float(np.mean([r["diverged"] for r in rows])),
        steps_survived_mean=float(np.mean(surv)),
        settling_time_mean=float(np.mean(settle)) if settle else float("nan"),
        control_effort_mean=float(np.mean([r["control_effort"] for r in rows])),
        peak_s_mean=float(np.mean([r["peak_s"] for r in rows])),
        peak_th1_mean=float(np.mean([r["peak_th1"] for r in rows])),
        peak_th2_mean=float(np.mean([r["peak_th2"] for r in rows])),
    )
    if steps:
        out["upright_to_end_rate"] = float(np.mean([s >= steps for s in surv]))
    return out


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


def make_mpc_policy(qf_lqr=False, cold=False, **mpc_kwargs):
    """The MPC teacher itself, as an evaluate_policy-compatible force policy.

    Exists so the teacher can be scored on the SAME metric as its students.
    Without this, a teacher that fails closed-loop is invisible -- which is
    exactly how a 1.0s horizon (C.MPC_NP=20 at C.DT=0.05) shipped: it stabilizes
    the angles and parks the cart at s_max, because recentring is infeasible
    inside the horizon, and every label inherits that.

    qf_lqr: use the infinite-horizon LQR cost-to-go as the terminal cost instead
    of the arbitrary Qf = 10*Q. A terminal cost that approximates the true
    value function is what lets a SHORT horizon behave like a long one.

    cold: re-solve from zeros every step instead of warm-starting from the
    shifted previous solution. Counter-intuitively better closed-loop: at
    Np=40, Qf=P it lifts centre-IC success 0.562 -> 0.750 -- the warm start
    keeps the solver in the basin of a previous, worse plan.
    """
    from mpc import MPCController

    def make_policy(mlparams):
        ml = np.asarray(mlparams)
        params = pu.full_params_from_ml(*ml)
        kw = dict(mpc_kwargs)
        if qf_lqr:
            kw["Qf"] = lqr_cost_to_go(ml, dt=C.DT)
        ctrl = MPCController(params, dt=C.DT, s_max=C.S_MAX, **kw)

        def policy(state):
            try:
                return ctrl.solve(state, cold=cold)[0]
            except RuntimeError:
                # Same warm-start reset dataset.py:80 uses after a failed solve:
                # a poisoned previous iterate makes every later solve fail too.
                ctrl._X_prev = np.zeros_like(ctrl._X_prev)
                ctrl._U_prev = np.zeros_like(ctrl._U_prev)
                return 0.0
        return policy
    return make_policy


def sample_interp_configs(n, rng, dataset_path=None):
    """Held-out configs inside the training (m,l) box -- same distribution, unseen combos.

    Callers MUST pass an rng offset by C.EVAL_SEED_OFFSET. The assert below is
    the guard: dataset generation draws from default_rng(C.SEED) into the same
    scrambled Sobol stream, so an un-offset rng reproduces training configs
    0..n-1 exactly -- which silently turns this split into a training-set
    measurement. Verified: it was 10/10 identical before the offset landed.
    """
    cfg = pu.sample_configs(n, rng=rng)
    try:
        seen = {tuple(np.round(r, 12)) for r in ds.load_dataset(dataset_path)["mlparams"]}
    except (FileNotFoundError, OSError):
        return cfg          # no dataset on disk yet -- nothing to leak from
    leaked = [tuple(c) for c in cfg if tuple(np.round(c, 12)) in seen]
    assert not leaked, (
        f"interpolation split leaked {len(leaked)}/{n} training configs "
        f"(e.g. {leaked[0]}) -- offset the eval rng by C.EVAL_SEED_OFFSET")
    return cfg


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
                 n_configs=10, n_ics=8, steps=150, seed=0, verbose=True,
                 epochs=None, ic_sampler=None):
    """
    Trains the three missing ablation variants (skips ones already checkpointed)
    then evaluates all four on the same held-out (interpolated) config set.
    Returns dict(variant -> summary dict).

    NOTE: pass a `full_ckpt` trained on the SAME dataset as the variants, or
    the comparison is confounded by the dataset rather than the loss terms.
    """
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    rng = np.random.default_rng(seed + C.EVAL_SEED_OFFSET)
    configs = (sample_interp_configs(n_configs, rng, dataset_path=dataset_path)
               if configs is None else configs)

    results = {}
    for name, overrides in ABLATIONS.items():
        if name == "full":
            ckpt = full_ckpt
        else:
            ckpt = os.path.join(ckpt_dir, f"ablation_{name}.pt")
            if not os.path.exists(ckpt):
                if verbose:
                    print(f"[ablation] training {name}...", flush=True)
                train(dataset_path=dataset_path, out_ckpt=ckpt, seed=seed,
                     verbose=verbose, epochs=epochs, weight_overrides=overrides)
        rows = evaluate_policy(make_pinn_policy(ckpt), configs, n_ics, steps,
                               seed=seed, ic_sampler=ic_sampler)
        results[name] = summarize(rows, steps=steps)
        results[name]["rows"] = rows      # keep per-rollout detail: not recoverable later,
                                          # since the leak fix changed the config stream
        if verbose:
            s = results[name]
            print(f"[ablation] {name:22s} success={s['success_rate']:.2f} "
                 f"upright={s.get('upright_to_end_rate', float('nan')):.2f} "
                 f"steps={s['steps_survived_mean']:.0f} "
                 f"effort={s['control_effort_mean']:.1f} "
                 f"peak_th1={s['peak_th1_mean']:.3f}", flush=True)
    return results


def run_generalization(full_ckpt, plain_ckpt=None, dataset_path=None, ckpt_dir=None,
                       n_configs=10, n_ics=8, steps=150, seed=0, verbose=True,
                       epochs=None, ic_sampler=None):
    """
    full PINN vs plain-imitation-NN vs LQR, each run on both an interpolation
    and an extrapolation held-out config split. Returns dict(split -> dict(controller -> summary)).
    """
    ckpt_dir = ckpt_dir or C.CKPT_DIR
    plain_ckpt = plain_ckpt or os.path.join(ckpt_dir, "ablation_data_only.pt")
    if not os.path.exists(plain_ckpt):
        if verbose:
            print("[generalization] training plain-imitation-NN baseline...", flush=True)
        # "Plain imitation" IS the data-only ablation -- same weight overrides.
        # Call train directly so the two can't drift apart: the previous
        # train_plain_imitation wrapper had no epochs= param, so this call
        # raised TypeError whenever --generalization ran without --ablation
        # (with --ablation first, the os.path.exists check above hid it).
        train(dataset_path=dataset_path, out_ckpt=plain_ckpt, seed=seed,
              verbose=verbose, epochs=epochs,
              weight_overrides=ABLATIONS["data_only"])

    rng = np.random.default_rng(seed + C.EVAL_SEED_OFFSET)
    splits = {
        "interpolation": sample_interp_configs(n_configs, rng, dataset_path=dataset_path),
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
            rows = evaluate_policy(make_policy, configs, n_ics, steps, seed=seed,
                                   ic_sampler=ic_sampler)
            results[split_name][ctrl_name] = summarize(rows, steps=steps)
            results[split_name][ctrl_name]["rows"] = rows
            if verbose:
                s = results[split_name][ctrl_name]
                print(f"[generalization] {split_name:14s} {ctrl_name:20s} "
                     f"success={s['success_rate']:.2f} "
                     f"upright={s.get('upright_to_end_rate', float('nan')):.2f} "
                     f"steps={s['steps_survived_mean']:.0f} "
                     f"effort={s['control_effort_mean']:.1f}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ckpt", default=os.path.join(C.CKPT_DIR, "spec60_full.pt"),
                    help="checkpoint for the 'full' variant. Do NOT pass round0_best.pt: it is "
                         "epoch 26 of 300, inside the data-only warmup, so it IS the data-only "
                         "net (making the ablation compare data-only to itself), and it carries "
                         "pre-60-tooth norm stats and a 173N actuator.")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--generalization", action="store_true")
    ap.add_argument("--mpc", action="store_true",
                    help="score the MPC TEACHER on this same metric. The teacher is the ceiling "
                         "for any student, so if it fails here nothing downstream can pass.")
    ap.add_argument("--mpc-np", type=int, default=None,
                    help="MPC horizon for --mpc. Deliberately NOT defaulted to config.MPC_NP so "
                         "it can be swept; omit to use the MPCController default.")
    ap.add_argument("--mpc-qf-lqr", action="store_true",
                    help="use the LQR cost-to-go as MPC terminal cost instead of Qf=10*Q")
    ap.add_argument("--mpc-cold", action="store_true",
                    help="cold-start every MPC solve (zeros) instead of warm-starting")
    ap.add_argument("--mpc-thdot-max", type=float, default=None,
                    help="angular-rate constraint for --mpc (mpc.py wires this but never passes it)")
    ap.add_argument("--n-configs", type=int, default=10)
    ap.add_argument("--n-ics", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=None,
                    help="epochs for any ablation/baseline training this triggers "
                         "(default config.EPOCHS=300, ~16 min per variant)")
    ap.add_argument("--dataset", default=None, help="dataset .npz for triggered training")
    ap.add_argument("--ckpt-dir", default=None,
                    help="where ablation_*.pt variants live (default pinn/checkpoints); "
                         "missing variants are trained there")
    ap.add_argument("--center-ics", action="store_true",
                    help="centre-only ICs instead of the off-centre/push mixture")
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help="seed for config draws and ICs (was silently always config.SEED)")
    ap.add_argument("--json-out", default=None,
                    help="override the results path; results are written either way")
    args = ap.parse_args()

    if not (args.ablation or args.generalization or args.mpc):
        args.ablation = args.generalization = True

    sampler = sample_center_states if args.center_ics else None
    common = dict(n_configs=args.n_configs, n_ics=args.n_ics, steps=args.steps,
                  epochs=args.epochs, dataset_path=args.dataset, ic_sampler=sampler,
                  seed=args.seed, ckpt_dir=args.ckpt_dir)

    def _git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=os.path.dirname(C.PACKAGE_DIR),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    # Provenance: without motor_force_max and git_commit, a result cannot be
    # attributed to a plant spec, which is how 173N-era numbers stayed quotable.
    out = {"config": {"full_ckpt": args.full_ckpt, "center_ics": args.center_ics,
                      "steps": args.steps, "n_configs": args.n_configs,
                      "n_ics": args.n_ics, "seed": args.seed, "epochs": args.epochs,
                      "dataset": args.dataset or C.SEED_DATASET,
                      "ckpt_dir": args.ckpt_dir or C.CKPT_DIR,
                      "motor_force_max": float(MOTOR_FORCE_MAX),
                      "mpc_np": args.mpc_np, "mpc_qf_lqr": args.mpc_qf_lqr,
                      "mpc_cold": args.mpc_cold,
                      "mpc_thdot_max": args.mpc_thdot_max,
                      "git_commit": _git_commit()}}

    if args.mpc:
        mpc_kw = {}
        if args.mpc_np is not None:
            mpc_kw["Np"] = args.mpc_np
        if args.mpc_thdot_max is not None:
            mpc_kw["thdot_max"] = args.mpc_thdot_max
        rng = np.random.default_rng(args.seed + C.EVAL_SEED_OFFSET)
        cfgs = sample_interp_configs(args.n_configs, rng, dataset_path=args.dataset)
        rows = evaluate_policy(make_mpc_policy(qf_lqr=args.mpc_qf_lqr, cold=args.mpc_cold, **mpc_kw),
                               cfgs, args.n_ics, args.steps, seed=args.seed,
                               ic_sampler=sampler)
        s = summarize(rows, steps=args.steps)
        s["rows"] = rows
        out["mpc_teacher"] = s
        print(f"[mpc teacher] Np={args.mpc_np or 'default'} qf_lqr={args.mpc_qf_lqr} "
              f"cold={args.mpc_cold} thdot_max={args.mpc_thdot_max}\n"
              f"  success={s['success_rate']:.3f} diverged={s['diverged_rate']:.3f} "
              f"steps={s['steps_survived_mean']:.0f}/{args.steps + 1} "
              f"settle={s['settling_time_mean']:.2f} peak_s={s['peak_s_mean']:.4f} "
              f"peak_th1={s['peak_th1_mean']:.4f}", flush=True)

    if args.ablation:
        out["ablation"] = run_ablation(args.full_ckpt, **common)
    if args.generalization:
        out["generalization"] = run_generalization(args.full_ckpt, **common)

    # Always persist. The 0%-vs-12% headline existed only as prose because this
    # was opt-in; an unwritten result is an unciteable one.
    path = args.json_out or os.path.join(
        C.RESULTS_DIR, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[evaluate] wrote {path}")
