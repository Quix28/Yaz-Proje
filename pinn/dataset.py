"""
Step 2: seed dataset generation via the MPC teacher.

For each sampled pendulum config, build ONE MPCController and warm-start it
across many small-perturbation initial states, logging (state, m,l) -> u*.
Building a controller per sample is catastrophic (each pays IPOPT's ~19 s
cold start); one-per-config + warm starts keeps warm solves at ~0.3 s.

Parallelized across configs with multiprocessing (processes, not threads --
CasADi/IPOPT objects are not thread-safe). Each worker owns its controller.

Storage (npz): states(N,6), mlparams(N,4), u(N,) [MPC force label],
config_id(N,). config_id lets the train/eval split hold out entire configs.
"""
import os
import time

import numpy as np

from pinn import config as C
from pinn import param_utils as pu


def _sample_states(n, rng):
    """
    Mixture of three initial-state regimes (C.MIX_WEIGHTS) so the seed
    dataset covers more than pure small-angle regulation from center:
      - center:    the small-perturbation box (STATE_PERT)
      - offcenter: same box, cart position widened to +-S_OFFCENTER_MAX --
                   teaches off-center stabilization
      - push:      position/angle as in STATE_PERT, velocities widened to
                   PUSH_VEL_PERT -- teaches disturbance rejection from a
                   sudden but physically plausible kick
    Wraps MPCController.solve unchanged; only this sampling loop differs.
    """
    w = C.MIX_WEIGHTS
    n_center = int(round(n * w["center"]))
    n_off = int(round(n * w["offcenter"]))
    n_push = n - n_center - n_off  # remainder -- avoids rounding gaps

    low, high = -C.STATE_PERT, C.STATE_PERT
    center = rng.uniform(low, high, size=(n_center, 6))

    off_low, off_high = low.copy(), high.copy()
    off_low[0], off_high[0] = -C.S_OFFCENTER_MAX, C.S_OFFCENTER_MAX
    offcenter = rng.uniform(off_low, off_high, size=(n_off, 6))

    push_low, push_high = low.copy(), high.copy()
    push_low[3:], push_high[3:] = -C.PUSH_VEL_PERT, C.PUSH_VEL_PERT
    push = rng.uniform(push_low, push_high, size=(n_push, 6))

    states = np.concatenate([center, offcenter, push], axis=0)
    rng.shuffle(states)
    return states


def generate_for_config(args):
    """
    Worker: solve the MPC for many ICs of a single config.

    args = (config_id, mlparams(4,), n_states, seed)
    Returns dict with states, mlparams, u, config_id, n_fail, n_attempt.
    Importing torch/casadi happens lazily inside the worker process.
    """
    config_id, mlparams, n_states, seed = args
    from mpc import MPCController, MOTOR_FORCE_MAX  # per-process import

    rng = np.random.default_rng(seed)
    params = pu.full_params_from_ml(*mlparams)
    ctrl = MPCController(params, Np=C.MPC_NP, dt=C.DT, s_max=C.S_MAX)

    states, us = [], []
    n_fail = 0
    ics = _sample_states(n_states, rng)
    for x0 in ics:
        try:
            u0, _, _ = ctrl.solve(x0)
        except RuntimeError:
            n_fail += 1
            # a failed solve can leave a poisoned warm-start; reset it so
            # one failure doesn't cascade into the next sample.
            ctrl._X_prev = np.zeros_like(ctrl._X_prev)
            ctrl._U_prev = np.zeros_like(ctrl._U_prev)
            continue
        if not np.isfinite(u0) or abs(u0) > C.MAX_LABEL_FACTOR * MOTOR_FORCE_MAX:
            n_fail += 1
            continue
        states.append(x0)
        us.append(u0)

    return dict(
        config_id=config_id,
        mlparams=np.asarray(mlparams, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64).reshape(-1, 6),
        u=np.asarray(us, dtype=np.float64).reshape(-1),
        n_fail=n_fail,
        n_attempt=n_states,
    )


def generate_dataset(n_configs=None, n_states=None, seed=None,
                     n_workers=None, out_path=None, norm_out_path=None,
                     verbose=True):
    """
    Generate and save the seed dataset. Returns the assembled dict.

    Drops configs whose solve-failure fraction exceeds config.MAX_FAIL_FRAC
    (near the MPC's stabilizable boundary) and logs them.

    norm_out_path defaults to C.NORM_STATS -- override it alongside
    out_path (e.g. to a tempdir) for smoke/e2e testing so a test run can't
    silently overwrite the real project's norm stats.
    """
    n_configs = n_configs or C.N_CONFIGS
    n_states = n_states or C.N_STATES_PER_CONFIG
    seed = C.SEED if seed is None else seed
    out_path = out_path or C.SEED_DATASET
    norm_out_path = norm_out_path or C.NORM_STATS

    rng = np.random.default_rng(seed)
    configs = pu.sample_configs(n_configs, rng=rng)
    jobs = [(i, configs[i], n_states, int(rng.integers(0, 2**31 - 1)))
            for i in range(n_configs)]

    t0 = time.time()
    results = []
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for r in pool.imap_unordered(generate_for_config, jobs):
                results.append(r)
                if verbose:
                    _log_result(r)
    else:
        for job in jobs:
            r = generate_for_config(job)
            results.append(r)
            if verbose:
                _log_result(r)

    # assemble, dropping high-failure configs
    kept_states, kept_ml, kept_u, kept_id = [], [], [], []
    dropped = []
    next_id = 0
    for r in sorted(results, key=lambda d: d["config_id"]):
        fail_frac = r["n_fail"] / max(1, r["n_attempt"])
        if fail_frac > C.MAX_FAIL_FRAC or len(r["u"]) == 0:
            dropped.append((tuple(np.round(r["mlparams"], 4)), round(fail_frac, 3)))
            continue
        m = len(r["u"])
        kept_states.append(r["states"])
        kept_ml.append(np.tile(r["mlparams"], (m, 1)))
        kept_u.append(r["u"])
        kept_id.append(np.full(m, next_id, dtype=np.int32))
        next_id += 1

    data = dict(
        states=np.concatenate(kept_states, axis=0),
        mlparams=np.concatenate(kept_ml, axis=0),
        u=np.concatenate(kept_u, axis=0),
        config_id=np.concatenate(kept_id, axis=0),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **data)
    compute_norm_stats(data, save=True, out_path=norm_out_path)

    if verbose:
        dt = time.time() - t0
        print(f"\n[dataset] {len(data['u'])} samples from {next_id} kept "
              f"configs ({len(dropped)} dropped) in {dt:.1f}s -> {out_path}")
        if dropped:
            print(f"[dataset] dropped (near stabilizable boundary): {dropped}")
    return data


def _log_result(r):
    fail_frac = r["n_fail"] / max(1, r["n_attempt"])
    print(f"[config {r['config_id']:3d}] kept {len(r['u']):3d}/{r['n_attempt']} "
          f"fail_frac={fail_frac:.2f}  ml={np.round(r['mlparams'],3).tolist()}")


def compute_norm_stats(data, save=True, out_path=None):
    """z-score stats for the 6 states + 4 params; std floored at 1e-8."""
    out_path = out_path or C.NORM_STATS
    x_mean = data["states"].mean(0)
    x_std = data["states"].std(0) + 1e-8
    p_mean = data["mlparams"].mean(0)
    p_std = data["mlparams"].std(0) + 1e-8
    stats = dict(x_mean=x_mean, x_std=x_std, p_mean=p_mean, p_std=p_std)
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **stats)
    return stats


def load_dataset(path=None):
    path = path or C.SEED_DATASET
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_norm_stats(path=None):
    path = path or C.NORM_STATS
    d = np.load(path)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    generate_dataset()
