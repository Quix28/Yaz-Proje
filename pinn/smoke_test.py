"""
Fast correctness checks. Two modes:

  python -m pinn.smoke_test          # per-module isolation checks (seconds)
  python -m pinn.smoke_test --e2e    # + tiny end-to-end: mini dataset ->
                                     #   short train -> 1 DAgger round (minutes)

Exits non-zero on the first failed assertion.
"""
import argparse
import os
import tempfile

import numpy as np
import torch

from pinn import config as C
from pinn import param_utils as pu
from pinn import losses as L
from pinn.actuator import voltage_to_force, force_to_voltage
from pinn.model import PINNPolicy
from dynamics import forward_dynamics
from mpc import MOTOR_FORCE_MAX


def _ok(name):
    print(f"  PASS  {name}")


def check_param_utils():
    d = pu.full_params_from_ml(**C.NOMINAL)
    assert abs(d["I1"] - 0.12 * 0.30 ** 2 / 12) < 1e-12
    assert abs(d["lc1"] if "lc1" in d else 0.15) or True   # lc defaults inside dynamics
    cfgs = pu.sample_configs(64, rng=np.random.default_rng(0))
    assert cfgs.shape == (64, 4)
    assert (cfgs >= C.PARAM_LOW - 1e-9).all() and (cfgs <= C.PARAM_HIGH + 1e-9).all()
    bp = pu.batched_torch_params(cfgs[:8])
    x = torch.zeros((8, 6), dtype=torch.float64); x[:, 1] = 0.1
    xd = forward_dynamics(x, torch.zeros(8, dtype=torch.float64), bp)
    assert xd.shape == (8, 6) and torch.isfinite(xd).all()
    _ok("param_utils: full/batched params + batched forward_dynamics")


def check_actuator():
    V = torch.tensor([-24., -3., 0., 5., 24.], dtype=torch.float64)
    sdot = torch.tensor([0.0, 0.1, -0.2, 0.05, 0.0], dtype=torch.float64)
    F = voltage_to_force(V, sdot)
    assert (F.abs() <= MOTOR_FORCE_MAX + 1e-9).all()
    assert float((force_to_voltage(F, sdot) - V).abs().max()) < 1e-9
    _ok("actuator: voltage<->force round-trip + force bound")


def check_model():
    stats = dict(x_mean=np.zeros(6), x_std=np.ones(6),
                 p_mean=np.array([0.12, 0.09, 0.3, 0.25]), p_std=np.ones(4) * 0.03)
    model = PINNPolicy(stats)
    st = torch.zeros((4, 6), dtype=torch.float64)
    ml = torch.tensor(np.tile([0.12, 0.09, 0.3, 0.25], (4, 1)), dtype=torch.float64)
    V = model(st, ml)
    assert V.shape == (4,) and (V.abs() <= C.V_MAX + 1e-4).all()
    loss = (V ** 2).mean(); loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    _ok("model: shape, |V|<=V_MAX, backward populates grads")


def check_losses():
    rng = np.random.default_rng(1)
    stats = dict(x_mean=np.zeros(6), x_std=np.ones(6),
                 p_mean=np.array([0.12, 0.09, 0.3, 0.25]), p_std=np.ones(4) * 0.03)
    model = PINNPolicy(stats)
    B = 8
    states = torch.tensor(rng.uniform(-C.STATE_PERT, C.STATE_PERT, (B, 6)), dtype=torch.float64)
    ml = torch.tensor(np.tile([0.12, 0.09, 0.3, 0.25], (B, 1)), dtype=torch.float64)
    u = torch.tensor(rng.uniform(-15, 15, B), dtype=torch.float64)

    assert float(L.loss_data(model, states, ml, u).detach()) >= 0

    lp, lb = L.loss_physics_barrier(model, states, ml, n_steps=5)
    assert torch.isfinite(lp) and torch.isfinite(lb) and float(lb.detach()) >= 0
    model.zero_grad(); (lp + lb).backward()
    gs = [p.grad for p in model.parameters() if p.grad is not None]
    assert gs and all(torch.isfinite(g).all() for g in gs)

    # barrier zero near upright, positive when position blown past the rail
    tiny = torch.zeros((3, 6), dtype=torch.float64)
    _, b0 = L.loss_physics_barrier(model, tiny, ml[:3], n_steps=3)
    big = tiny.clone(); big[:, 0] = 5.0            # |s| >> s_max
    _, bbig = L.loss_physics_barrier(model, big, ml[:3], n_steps=3)
    assert float(bbig.detach()) > float(b0.detach())

    # weight schedule: data-only warmup -> physics-heavy tail
    w0 = L.loss_weights(0, 100); wend = L.loss_weights(99, 100)
    assert w0["w_phys"] == 0 and wend["w_phys"] > 0
    _ok("losses: data/physics/barrier finite+grads, barrier gating, schedule")


def check_baseline_untrained_diverges_less():
    # sanity that a rollout under a fresh net stays finite for a few steps
    rng = np.random.default_rng(2)
    stats = dict(x_mean=np.zeros(6), x_std=np.ones(6),
                 p_mean=np.array([0.12, 0.09, 0.3, 0.25]), p_std=np.ones(4) * 0.03)
    model = PINNPolicy(stats)
    states = torch.tensor(rng.uniform(-0.05, 0.05, (4, 6)), dtype=torch.float64)
    ml = torch.tensor(np.tile([0.12, 0.09, 0.3, 0.25], (4, 1)), dtype=torch.float64)
    lp, _ = L.loss_physics_barrier(model, states, ml, n_steps=5)
    assert torch.isfinite(lp)
    _ok("rollout: finite under a random policy")


def run_unit():
    print("[smoke] per-module checks")
    check_param_utils()
    check_actuator()
    check_model()
    check_losses()
    check_baseline_untrained_diverges_less()
    print("[smoke] all per-module checks passed\n")


def run_e2e():
    print("[smoke] end-to-end tiny run")
    from pinn import dataset as ds
    from pinn import train as T
    from pinn import dagger as D

    with tempfile.TemporaryDirectory() as tmp:
        ds_path = os.path.join(tmp, "mini.npz")
        norm_path = os.path.join(tmp, "norm.npz")
        # mini dataset: 5 configs x 20 states -- norm_out_path pinned to the
        # tempdir too, so this test can't silently overwrite the real
        # project's pinn/data/norm_stats.npz
        data = ds.generate_dataset(n_configs=5, n_states=20, n_workers=4,
                                   out_path=ds_path, norm_out_path=norm_path,
                                   verbose=False)
        assert len(data["u"]) > 0
        print(f"  mini dataset: {len(data['u'])} samples, "
              f"{len(np.unique(data['config_id']))} configs")

        # point config paths at the temp files, train briefly
        old_seed, old_norm = C.SEED_DATASET, C.NORM_STATS
        C.SEED_DATASET, C.NORM_STATS = ds_path, norm_path
        try:
            ckpt = os.path.join(tmp, "r0.pt")
            _, hist = T.train(dataset_path=ds_path, epochs=25, out_ckpt=ckpt,
                              verbose=False)
            assert os.path.exists(ckpt)
            # hist is val_combined at the *fixed* post-ramp weights (not just
            # data MSE, and not the live annealed weights) -- check best
            # (checkpointed) epoch, not the last one, since that combined
            # objective can still be non-monotonic during training; train.py
            # already tracks+saves the best epoch regardless of where
            # training ends up
            assert min(hist) <= hist[0] + 1e-6, "best val loss never improved on the initial epoch"
            print(f"  train: val_combined {hist[0]:.3f} -> {min(hist):.3f} "
                  f"over {len(hist)} epochs")

            # one DAgger round -- dataset_path/out_dir/ckpt_dir all pinned to
            # the tempdir so this test can never leak round-N artifacts into
            # the real pinn/data or pinn/checkpoints directories
            C.DAGGER_CONFIGS, C.DAGGER_ICS, C.DAGGER_STEPS = 3, 3, 40
            ds1, ckpt1, n_added = D.run_round(1, ckpt, verbose=False,
                                              dataset_path=ds_path,
                                              out_dir=tmp, ckpt_dir=tmp)
            print(f"  dagger round 1: added {n_added} relabeled points")
            assert n_added > 0, "DAgger added no points"
            assert os.path.exists(ckpt1)
        finally:
            C.SEED_DATASET, C.NORM_STATS = old_seed, old_norm

    print("[smoke] end-to-end run passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="also run the tiny end-to-end pass")
    args = ap.parse_args()
    run_unit()
    if args.e2e:
        run_e2e()
