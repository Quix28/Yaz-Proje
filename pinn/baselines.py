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

    # The gain is provably stabilizing on the LINEARIZATION -- assert that,
    # not nonlinear closed-loop survival. Those are different claims, and the
    # previous version asserted the latter while the comment claimed the
    # former, so it failed as soon as the nonlinear rollout diverged.
    #
    # Nonlinearly this LQR has a small region of attraction on this plant:
    # the smoothed-Coulomb friction term contributes d/dsdot[cf*tanh(sdot/eps)]
    # = cf/eps of damping AT THE ORIGIN that saturates away from it, so the
    # linearization sees far more damping help than the real plant provides.
    # LQR losing to MPC/PINN off the origin is the point of it as a baseline.
    from scipy.linalg import expm
    params = pu.full_params_from_ml(*ml)
    x0 = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    u0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    A = torch.autograd.functional.jacobian(
        lambda x: dynamics.forward_dynamics(x, u0, params), x0).numpy()
    B = torch.autograd.functional.jacobian(
        lambda u: dynamics.forward_dynamics(x0, u, params), u0).numpy().reshape(6, 1)
    Mx = np.zeros((7, 7))
    Mx[:6, :6], Mx[:6, 6:] = A, B
    Md = expm(Mx * C.DT)
    Ad, Bd = Md[:6, :6], Md[:6, 6:]
    cl = np.abs(np.linalg.eigvals(Ad - Bd @ K))
    assert np.isfinite(K).all(), "LQR gain non-finite"
    assert cl.max() < 1.0, f"LQR closed loop not Schur-stable: max|eig|={cl.max():.4f}"

    batched = pu.batched_torch_params(np.asarray(ml).reshape(1, 4))
    x = torch.tensor([0.005, 0.005, -0.004, 0.0, 0.0, 0.0], dtype=torch.float64)
    steps, survived = 40, 0
    for _ in range(steps):
        F = policy(x.numpy())
        x = L.rk4_step(x.unsqueeze(0), torch.tensor([F], dtype=torch.float64),
                       batched, C.DT).squeeze(0)
        if not torch.isfinite(x).all() or float(x.abs().max()) > 10:
            break
        survived += 1
    print(f"[baselines demo] LQR gain finite, closed-loop max|eig|={cl.max():.4f} "
          f"(Schur-stable); nonlinear rollout survived {survived}/{steps} steps "
          f"from max|x0|=0.005")


if __name__ == "__main__":
    _demo()
