"""
Step 4: loss terms + annealing schedule.

  L_data    imitation MSE, in VOLTAGE (MPC force labels -> voltage targets)
  L_physics differentiable N-step rollout through forward_dynamics under the
            network's own commands; penalize deviation from upright over the
            whole rollout, per-sample under each sample's own pendulum.
  L_barrier soft one-sided penalties for position/velocity/force-rate limits.
  L_EL      Euler-Lagrange / Lyapunov residual at random collocation points
            (state,param combos the MPC never solved) -- the "real PINN" term.

Precision: the network path is float32; state + force are cast to float64
before entering forward_dynamics (mass-matrix inverse is float64-stable),
and the scalar loss is used as-is by autograd across the cast.
"""
import numpy as np
import torch

from pinn import config as C
from pinn import param_utils as pu
from pinn.actuator import voltage_to_force, force_to_voltage
from dynamics import forward_dynamics
from mpc import MOTOR_FREE_SPEED

_F64 = torch.float64
_STATE_W = torch.tensor(C.STATE_COST_W, dtype=_F64)
DATA_FLOOR = 0.3   # w_data at the end of the ramp (see schedule table)


def rk4_step(state, u, params, dt):
    """One RK4 step of the torch plant (zero-order hold on u across substeps).
    Mirrors sim_loop._rk4_step_torch and mpc._rk4_step exactly."""
    k1 = forward_dynamics(state, u, params)
    k2 = forward_dynamics(state + dt / 2 * k1, u, params)
    k3 = forward_dynamics(state + dt / 2 * k2, u, params)
    k4 = forward_dynamics(state + dt * k3, u, params)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def loss_data(model, states, mlparams, u_labels):
    """MSE between predicted voltage and the MPC label converted to voltage."""
    sdot = states[:, 3]
    v_target = force_to_voltage(u_labels, sdot)          # (B,) voltage
    v_pred = model(states, mlparams)                     # (B,) voltage
    return torch.mean((v_pred - v_target.to(v_pred.dtype)) ** 2)


def _rollout(model, x0, mlparams, dt, n_steps):
    """
    Shared N-step rollout used by both physics and barrier losses.

    Returns (dev_accum, barrier_accum): mean-over-rollout weighted deviation
    from upright, and mean-over-rollout constraint-violation penalty. Each
    batch element rolls out under its own pendulum via batched params.
    """
    x = x0.to(_F64)
    batched = pu.batched_torch_params(mlparams, dtype=_F64)
    w = _STATE_W.to(x.device)

    s_max = C.S_MAX
    sdot_max = MOTOR_FREE_SPEED
    du_max = C.DU_MAX

    dev = x.new_zeros(())
    barrier = x.new_zeros(())
    u_prev = None
    for _ in range(n_steps):
        V = model(x, mlparams).to(_F64)                  # voltage
        F = voltage_to_force(V, x[:, 3])                 # force [N]
        x = rk4_step(x, F, batched, dt)

        dev = dev + torch.mean(torch.sum(w * x ** 2, dim=-1))

        pen_s = torch.relu(x[:, 0].abs() - s_max) ** 2
        pen_v = torch.relu(x[:, 3].abs() - sdot_max) ** 2
        pen = pen_s + pen_v
        if u_prev is not None:
            pen = pen + torch.relu((F - u_prev).abs() - du_max) ** 2
        barrier = barrier + torch.mean(pen)
        u_prev = F

    return dev / n_steps, barrier / n_steps


def loss_physics_barrier(model, states, mlparams, dt=None, n_steps=None):
    """Roll out once; return (L_physics, L_barrier) (they share the rollout)."""
    dt = C.DT if dt is None else dt
    n_steps = C.PHYS_N if n_steps is None else n_steps
    return _rollout(model, states, mlparams, dt, n_steps)


def loss_el(model, rng, n=None):
    """
    Euler-Lagrange / Lyapunov residual at random collocation points from a
    WIDER (state,param) box than the dataset -- regions the MPC never labels.
    Penalize commands that physically accelerate a link *away* from upright:
    with theta measured from vertical-up, a restoring command makes
    theta*theta_ddot < 0, so relu(theta*theta_ddot) is the violation.
    Reuses only forward_dynamics (no MPC, cheap).
    """
    n = C.N_COLLOC if n is None else n
    ml = pu.sample_configs(n, rng=rng,
                           low=C.COLLOC_PARAM_LOW, high=C.COLLOC_PARAM_HIGH)
    pert = C.COLLOC_STATE_PERT
    x = rng.uniform(-pert, pert, size=(n, 6))
    x_t = torch.tensor(x, dtype=_F64)
    ml_t = torch.tensor(ml, dtype=_F64)

    V = model(x_t, ml_t).to(_F64)
    F = voltage_to_force(V, x_t[:, 3])
    batched = pu.batched_torch_params(ml_t, dtype=_F64)
    xdot = forward_dynamics(x_t, F, batched)

    th1, th2 = x_t[:, 1], x_t[:, 2]
    th1dd, th2dd = xdot[:, 4], xdot[:, 5]
    residual = torch.relu(th1 * th1dd) + torch.relu(th2 * th2dd)
    return torch.mean(residual) + C.EL_EPS * torch.mean(V ** 2)


def loss_weights(epoch, epochs=None):
    """
    Annealing schedule: data-only warmup -> ramp -> physics/barrier-heavy.
    Returns dict(w_data, w_phys, w_bar, w_el).
    """
    epochs = C.EPOCHS if epochs is None else epochs
    frac = epoch / max(1, epochs - 1)
    warm, ramp = C.WARMUP_FRAC, C.RAMP_FRAC

    if frac <= warm:
        return dict(w_data=C.W_DATA, w_phys=0.0, w_bar=0.0, w_el=0.0)
    if frac >= ramp:
        return dict(w_data=DATA_FLOOR, w_phys=C.W_PHYS, w_bar=C.W_BAR, w_el=C.W_EL)
    # linear interpolation across the ramp
    t = (frac - warm) / (ramp - warm)
    return dict(
        w_data=C.W_DATA + t * (DATA_FLOOR - C.W_DATA),
        w_phys=t * C.W_PHYS,
        w_bar=t * C.W_BAR,
        w_el=t * C.W_EL,
    )


def combined_loss(model, batch, epoch, rng, epochs=None):
    """
    Full weighted loss for one batch. batch = (states, mlparams, u_labels)
    as float tensors. Physics terms only computed when their weight > 0
    (skips the rollout entirely during data-only warmup).
    Returns (total, components_dict).
    """
    states, mlparams, u_labels = batch
    w = loss_weights(epoch, epochs)

    l_data = loss_data(model, states, mlparams, u_labels)
    total = w["w_data"] * l_data
    comps = {"data": float(l_data.detach())}

    if w["w_phys"] > 0 or w["w_bar"] > 0:
        l_phys, l_bar = loss_physics_barrier(model, states, mlparams)
        total = total + w["w_phys"] * l_phys + w["w_bar"] * l_bar
        comps["phys"] = float(l_phys.detach())
        comps["bar"] = float(l_bar.detach())
    if w["w_el"] > 0:
        l_el = loss_el(model, rng)
        total = total + w["w_el"] * l_el
        comps["el"] = float(l_el.detach())

    comps["weights"] = w
    return total, comps
