"""
Step 6 baselines: LQR (closed-form, linearized about the upright
equilibrium) and the plain-imitation-NN ablation -- CLAUDE.md Step 6 calls
for both as comparison points against the full PINN.

The "plain imitation NN" baseline IS the data-only ablation variant
(train.py's weight_overrides={'w_phys':0,'w_bar':0,'w_el':0}) -- same
network/training loop, physics/barrier/EL terms just pinned off, so it's
reused for both the ablation table and the generalization baseline.
"""
import numpy as np
import torch
from scipy.linalg import expm, solve_discrete_are

import dynamics
from pinn import config as C
from pinn import param_utils as pu
from mpc import MOTOR_FORCE_MAX


def lqr_gain(mlparams, dt=None, Q=None, R=1e-2):
    """
    Discrete-time LQR gain linearized about the upright equilibrium
    (x=0, u=0): autograd jacobians of forward_dynamics + exact
    zero-order-hold discretization (matrix exponential). u = -K @ x [N].
    """
    dt = dt or C.DT
    params = pu.full_params_from_ml(*mlparams)
    Qm = np.diag(C.STATE_COST_W) if Q is None else np.asarray(Q, dtype=np.float64)
    Rm = np.atleast_2d(np.asarray(R, dtype=np.float64))

    x0 = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    u0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    A = torch.autograd.functional.jacobian(
        lambda x: dynamics.forward_dynamics(x, u0, params), x0).numpy()
    B = torch.autograd.functional.jacobian(
        lambda u: dynamics.forward_dynamics(x0, u, params), u0).numpy().reshape(6, 1)

    n = 6
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n:] = A, B
    Md = expm(M * dt)
    Ad, Bd = Md[:n, :n], Md[:n, n:]

    P = solve_discrete_are(Ad, Bd, Qm, Rm)
    K = np.linalg.solve(Rm + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K


def lqr_policy(K):
    """force-output policy: state (6,) ndarray -> saturated force [N]."""
    def policy(state):
        u = float(-(K @ np.asarray(state))[0])
        return max(-MOTOR_FORCE_MAX, min(MOTOR_FORCE_MAX, u))
    return policy


def train_plain_imitation(dataset_path=None, out_ckpt=None, init_ckpt=None,
                          seed=None, verbose=True, use_wandb=False):
    """Data-only ablation == the 'plain imitation NN' generalization baseline."""
    from pinn.train import train
    return train(dataset_path=dataset_path, out_ckpt=out_ckpt, init_ckpt=init_ckpt,
                seed=seed, verbose=verbose, use_wandb=use_wandb,
                weight_overrides={"w_phys": 0.0, "w_bar": 0.0, "w_el": 0.0})


def _demo():
    from pinn import losses as L
    ml = (C.NOMINAL["m1"], C.NOMINAL["m2"], C.NOMINAL["l1"], C.NOMINAL["l2"])
    K = lqr_gain(ml)
    policy = lqr_policy(K)

    batched = pu.batched_torch_params(np.asarray(ml).reshape(1, 4))
    x = torch.tensor([0.05, 0.1, -0.08, 0.0, 0.0, 0.0], dtype=torch.float64)
    x0_norm = float(x.abs().max())
    for _ in range(200):
        F = policy(x.numpy())
        x = L.rk4_step(x.unsqueeze(0), torch.tensor([F], dtype=torch.float64),
                       batched, C.DT).squeeze(0)
        assert torch.isfinite(x).all(), "LQR rollout diverged (non-finite state)"
    xT_norm = float(x.abs().max())
    assert xT_norm < x0_norm, f"LQR failed to stabilize: {x0_norm:.4f} -> {xT_norm:.4f}"
    print(f"[baselines demo] LQR stabilized nominal config: "
          f"max|x0|={x0_norm:.4f} -> max|xT|={xT_norm:.4f} over 200 steps")


if __name__ == "__main__":
    _demo()
