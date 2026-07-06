"""
Two-phase controller: energy-shaping swing-up, then handoff to the
existing MPC (mpc.py) for catch + stabilize near upright.

Why two phases: confirmed empirically (energy_budget check) the motor
has plenty of energy headroom, but a single short-horizon MPC can't
"see" a multi-second pumping strategy and just gives up at the bottom.
Classic fix -- an energy-shaping swing-up law to get near vertical,
then switch to MPC/LQR for the final capture, which we already
validated works well from small perturbations.

Energy relation used (exact, from the Lagrangian): u only does work
through the cart's own coordinate, so dE/dt = u*sdot - damping losses,
regardless of how the two links are coupled. Pump rule: push in the
direction of cart motion to add energy, oppose it to remove energy --
no double-pendulum-specific heuristic needed. Rail-limit avoidance
overrides this near the physical ends of travel.
"""
import numpy as np
import torch

from dynamics import forward_dynamics
from mpc import MPCController, MOTOR_FORCE_MAX, MOTOR_FREE_SPEED


def _mass_matrix(th1, th2, p):
    m1, m2, l1, l2 = p['m1'], p['m2'], p['l1'], p['l2']
    I1, I2, M = p['I1'], p['I2'], p['M']
    lc1, lc2 = p.get('lc1', l1 / 2), p.get('lc2', l2 / 2)
    m_enc1, m_enc2 = p.get('m_enc1', 0.18), p.get('m_enc2', 0.18)
    c1, c2 = np.cos(th1), np.cos(th2)
    c12 = np.cos(th1 - th2)
    M_eff = M + m_enc1
    m2_l1 = m2 + m_enc2
    M11 = M_eff + m1 + m2_l1
    M12 = (m1 * lc1 + m2_l1 * l1) * c1
    M13 = m2 * lc2 * c2
    M22 = m1 * lc1 ** 2 + m2_l1 * l1 ** 2 + I1
    M23 = m2 * l1 * lc2 * c12
    M33 = m2 * lc2 ** 2 + I2
    return np.array([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])


def _potential(th1, th2, p):
    m1, m2, l1 = p['m1'], p['m2'], p['l1']
    lc1, lc2 = p.get('lc1', p['l1'] / 2), p.get('lc2', p['l2'] / 2)
    g = p.get('g', 9.81)
    m_enc2 = p.get('m_enc2', 0.18)
    m2_l1 = m2 + m_enc2
    return m1 * g * lc1 * np.cos(th1) + m2_l1 * g * l1 * np.cos(th1) + m2 * g * lc2 * np.cos(th2)


def energy(state, p):
    s, th1, th2, sdot, th1dot, th2dot = state
    qdot = np.array([sdot, th1dot, th2dot])
    Mmat = _mass_matrix(th1, th2, p)
    KE = 0.5 * qdot @ Mmat @ qdot
    return KE + _potential(th1, th2, p)


def wrapped_angle(theta):
    """distance-to-nearest-upright representation, in (-pi, pi]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


SPEED_GOVERNOR_FRAC = 0.85  # keep meaningful force margin (~28% of F_max
                            # still available here) instead of letting
                            # commanded speed brush the free-speed
                            # ceiling, where force -- and therefore any
                            # ability to recover -- is exactly zero


def swingup_control(state, p, s_max, E_target, k_E=300.0, k_center=5.0):
    """
    Continuous energy-shaping pump: u ~ -k_E*(E-E_target)*sdot, saturated
    to the speed-derated available force. This naturally backs off as E
    approaches E_target (unlike a fixed-magnitude bang-bang pump, which
    keeps demanding max force in the current direction of motion and
    drives cart speed past the motor's free-speed ceiling with no way
    back once force derates to zero there).

    Two safety layers:
    1. Speed governor at SPEED_GOVERNOR_FRAC*free-speed -- leaves a real
       margin (unlike an earlier too-tight 0.6 cap that killed pump
       amplitude, and unlike no cap at all, which let a strong pump
       overshoot into the exact-zero-force region near free-speed).
    2. Predictive rail brake, sized off the force available *at the
       governor ceiling* (not the instantaneous, possibly near-zero,
       force right as the ceiling is approached -- that was
       self-referential and underestimated stopping distance).
    """
    s, th1, th2, sdot, th1dot, th2dot = state

    u_avail = MOTOR_FORCE_MAX * max(0.0, 1 - (sdot / MOTOR_FREE_SPEED) ** 2)
    v_cap = SPEED_GOVERNOR_FRAC * MOTOR_FREE_SPEED

    if abs(sdot) > v_cap:
        return -np.sign(sdot) * u_avail

    total_mass = p['M'] + p['m1'] + p['m2']
    a_brake_ref = MOTOR_FORCE_MAX * (1 - SPEED_GOVERNOR_FRAC ** 2) / total_mass
    stop_dist = sdot ** 2 / (2 * a_brake_ref)
    if sdot > 0 and s + stop_dist > s_max:
        return -u_avail
    if sdot < 0 and s - stop_dist < -s_max:
        return u_avail

    E = energy(state, p)
    u_pump = -k_E * (E - E_target) * sdot
    u_center = -k_center * s  # gentle continuous centering bias, avoids drift
    u = np.clip(u_pump + u_center, -u_avail, u_avail)
    return u


def near_upright(state, angle_tol=0.45, rate_tol=2.0):
    _, th1, th2, _, th1dot, th2dot = state
    return (abs(wrapped_angle(th1)) < angle_tol
            and abs(wrapped_angle(th2)) < angle_tol
            and abs(th1dot) < rate_tol and abs(th2dot) < rate_tol)


def _rk4(state, u, params, dt):
    st = torch.tensor(state, dtype=torch.float64)
    ut = torch.tensor(float(u), dtype=torch.float64)
    k1 = forward_dynamics(st, ut, params).numpy()
    k2 = forward_dynamics(torch.tensor(state + dt / 2 * k1), ut, params).numpy()
    k3 = forward_dynamics(torch.tensor(state + dt / 2 * k2), ut, params).numpy()
    k4 = forward_dynamics(torch.tensor(state + dt * k3), ut, params).numpy()
    new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # hard velocity safety clamp -- stand-in for a real driver's
    # current-limit/stall protection, prevents runaway drift if any
    # controller imperfection pushes speed past what the motor can do
    new_state[3] = np.clip(new_state[3], -1.05 * MOTOR_FREE_SPEED, 1.05 * MOTOR_FREE_SPEED)
    return new_state


def run(params, x0, swingup_dt=0.002, mpc_dt=0.05, swingup_time=20.0,
        mpc_steps=300, s_max=0.18, Np=20):
    """
    Phase 1: energy-shaping swing-up. Uses a much finer dt than the MPC
    phase -- at max acceleration (~F_max/mass) a single 0.02s-0.05s step
    can swing cart velocity clean through the entire free-speed range
    before this simple reactive law gets a chance to respond. The MPC
    phase gets away with a slower loop because IPOPT plans several steps
    ahead inside the solve; this reactive law needs a much faster loop,
    same as a real embedded controller would run.

    Phase 2: handoff to MPCController once near upright, xref set to the
    nearest 2*pi-equivalent of current (unwrapped) angles -- MPC's cost
    doesn't know about periodicity, so we give it a reachable target
    instead of literal zero if the swing-up wound through extra rotations.
    """
    E_target = _potential(0.0, 0.0, params)
    state = np.asarray(x0, dtype=np.float64)
    log = []  # (t, s, th1, th2, sdot, th1dot, th2dot, u, mode)
    t = 0.0

    # both links start exactly at the stable equilibrium (zero velocity,
    # zero net torque) -- an exact fixed point, so a tiny priming kick is
    # needed to break symmetry before the energy-shaping law has anything
    # (nonzero sdot) to act on.
    PRIME_STEPS = int(round(0.2 / swingup_dt))
    PRIME_FORCE = 0.3 * MOTOR_FORCE_MAX

    swingup_steps = int(round(swingup_time / swingup_dt))
    handed_off = False
    for i in range(swingup_steps):
        if near_upright(state):
            handed_off = True
            break
        if i < PRIME_STEPS:
            u = PRIME_FORCE
        else:
            u = swingup_control(state, params, s_max, E_target)
        state = _rk4(state, u, params, swingup_dt)
        t += swingup_dt
        log.append((t, *state.tolist(), u, "swingup"))

    if not handed_off:
        return log, False  # swing-up didn't reach catch basin in time

    th1_ref = round(state[1] / (2 * np.pi)) * 2 * np.pi
    th2_ref = round(state[2] / (2 * np.pi)) * 2 * np.pi
    xref = np.array([0.0, th1_ref, th2_ref, 0.0, 0.0, 0.0])

    ctrl = MPCController(params, Np=Np, dt=mpc_dt, s_max=s_max)
    for i in range(mpc_steps):
        u0, X_pred, _ = ctrl.solve(state, xref=xref)
        state = _rk4(state, u0, params, mpc_dt)
        t += mpc_dt
        log.append((t, *state.tolist(), u0, "mpc"))

    return log, True


if __name__ == "__main__":
    params = dict(
        # m1, m2 are rod-only mass -- encoder mass (180g each) tracked
        # separately via m_enc1 (on the cart) / m_enc2 (at the joint,
        # rotates with th1 only)
        m1=0.12, m2=0.09, l1=0.3, l2=0.25, M=1.0,
        m_enc1=0.18, m_enc2=0.18,
        I1=0.12 * 0.3 ** 2 / 12, I2=0.09 * 0.25 ** 2 / 12,
    )
    x0 = [0.0, np.pi, np.pi, 0.0, 0.0, 0.0]  # both links hanging down

    log, caught = run(params, x0, swingup_dt=0.002, mpc_dt=0.05,
                       swingup_time=20.0, mpc_steps=300, s_max=0.18)
    print(f"caught: {caught}, total steps logged: {len(log)}")

    max_s = max(abs(row[1]) for row in log)
    print(f"max |s| over run: {max_s:.4f}")

    for row in log[::200]:
        t, s, th1, th2, sdot, th1dot, th2dot, u, mode = row
        print(f"t={t:6.2f}s  s={s:7.3f}  th1={th1:7.3f}  th2={th2:7.3f}  "
              f"u={u:7.3f}  [{mode}]")
    print("...")
    for row in log[-15:]:
        t, s, th1, th2, sdot, th1dot, th2dot, u, mode = row
        print(f"t={t:6.2f}s  s={s:7.3f}  th1={th1:7.3f}  th2={th2:7.3f}  "
              f"u={u:7.3f}  [{mode}]")
