"""
Parameter sampling and param-dict construction.

The network is conditioned on (m1, m2, l1, l2). Everything else the physics
needs -- moments of inertia, COM distances, cart mass, encoder masses -- is
either derived from those four or held fixed. Two consumers:

  * dataset generation / MPC teacher: needs plain-float param dicts.
  * L_physics rollout: needs per-sample params as (B,) tensors so each
    batch element rolls out under its own pendulum (forward_dynamics
    broadcasts params against state[...,i] of shape (B,)).
"""
import numpy as np
import torch

from pinn import config as C

try:                                  # even coverage of the 4-D param box
    from scipy.stats import qmc
    _HAVE_QMC = True
except Exception:                     # pragma: no cover - fallback
    _HAVE_QMC = False


def sample_configs(n, rng=None, low=None, high=None):
    """
    Sample n configs from the (m1,m2,l1,l2) box.

    Returns (n, 4) float64 array. Uses a scrambled Sobol sequence for even
    coverage when scipy is available (clumpy uniform sampling hurts the
    generalization story), else stratified uniform.
    """
    if low is None:
        low = C.PARAM_LOW
    if high is None:
        high = C.PARAM_HIGH
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    if _HAVE_QMC:
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
        unit = sampler.random(n)                     # (n,4) in [0,1)
    else:
        rng = np.random.default_rng() if rng is None else rng
        unit = rng.uniform(size=(n, 4))
    return low + unit * (high - low)


def full_params_from_ml(m1, m2, l1, l2):
    """
    Build the complete params dict forward_dynamics / MPCController expect
    from the four conditioning values. Dependents follow the uniform-rod
    convention used everywhere else: I_i = m_i*l_i**2/12, lc_i = l_i/2.
    Encoder masses and cart mass are fixed (config).
    """
    m1, m2, l1, l2 = float(m1), float(m2), float(l1), float(l2)
    return dict(
        m1=m1, m2=m2, l1=l1, l2=l2,
        M=C.CART_MASS,
        m_enc1=C.M_ENC1, m_enc2=C.M_ENC2,
        I1=m1 * l1 ** 2 / 12.0,
        I2=m2 * l2 ** 2 / 12.0,
    )


def batched_torch_params(mlparams, device="cpu", dtype=torch.float64):
    """
    mlparams: (B,4) array/tensor of [m1,m2,l1,l2].

    Returns a params dict whose values are (B,) float64 tensors, ready to
    pass straight into forward_dynamics for a batched, per-sample rollout.
    Fixed params (M, encoders) are broadcast to (B,); dependents recomputed.
    """
    p = torch.as_tensor(mlparams, dtype=dtype, device=device)
    if p.ndim == 1:
        p = p.unsqueeze(0)
    m1, m2, l1, l2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    ones = torch.ones_like(m1)
    return dict(
        m1=m1, m2=m2, l1=l1, l2=l2,
        M=C.CART_MASS * ones,
        m_enc1=C.M_ENC1 * ones, m_enc2=C.M_ENC2 * ones,
        I1=m1 * l1 ** 2 / 12.0,
        I2=m2 * l2 ** 2 / 12.0,
    )
