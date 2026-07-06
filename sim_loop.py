"""
Closed-loop sim: MPC (mpc.py, CasADi/IPOPT) senses state, solves, applies
only u0. Plant stepped independently via dynamics.py (torch RK4) -- a
separate "physics engine" from the RK4 model baked inside the solver's
NLP. Today both use identical equations/params, so no mismatch shows up;
this is where you'd later inject real-rig parameter error, sensor noise,
or unmodeled friction without touching the controller.

Sense -> Solve -> Apply u0 -> Step plant -> Shift horizon -> repeat.
"""
import numpy as np
import torch

from dynamics import forward_dynamics
from mpc import MPCController, MOTOR_FORCE_MAX


def _rk4_step_torch(state, u, params, dt):
    k1 = forward_dynamics(state, u, params)
    k2 = forward_dynamics(state + dt / 2 * k1, u, params)
    k3 = forward_dynamics(state + dt / 2 * k2, u, params)
    k4 = forward_dynamics(state + dt * k3, u, params)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run(params, x0, Np=20, dt=0.05, steps=120, s_max=0.18, u_max=MOTOR_FORCE_MAX):
    ctrl = MPCController(params, Np=Np, dt=dt, s_max=s_max, u_max=u_max)

    x = np.asarray(x0, dtype=np.float64)
    log = []

    for i in range(steps):
        u0, X_pred, U_pred = ctrl.solve(x)          # sense x(t) -> solve

        x_t = torch.tensor(x, dtype=torch.float64)
        u_t = torch.tensor(u0, dtype=torch.float64)
        x_next = _rk4_step_torch(x_t, u_t, params, dt)  # apply u0 -> step plant

        x = x_next.numpy()
        log.append((i, *x.tolist(), u0))

    return log


if __name__ == "__main__":
    params = dict(
        m1=0.2, m2=0.15, l1=0.3, l2=0.25, M=1.0,
        I1=0.2 * 0.3**2 / 12, I2=0.15 * 0.25**2 / 12,
    )
    x0 = [0.0, np.pi, np.pi, 0.0, 0.0, 0.0]  # both links hanging down

    log = run(params, x0, Np=20, dt=0.05, steps=120)

    print(f"{'step':>4} {'s':>8} {'th1':>8} {'th2':>8} {'u':>8}")
    for row in log:
        i, s, th1, th2, sdot, th1dot, th2dot, u = row
        if i % 5 == 0 or i > 100:
            print(f"{i:4d} {s:8.4f} {th1:8.4f} {th2:8.4f} {u:8.3f}")
