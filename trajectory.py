"""
Offline swing-up trajectory generation via direct collocation (CasADi +
IPOPT), boundary-value formulation -- not a receding-horizon controller.

Why this over the receding-horizon MPC (mpc.py) or the energy-shaping
heuristic (swingup.py) for the swing-up itself: both got stuck. The
short-horizon MPC can't see far enough ahead to value a multi-second
pumping maneuver. The energy-shaping heuristic pumps *total* energy
toward the upright value, but matching total energy doesn't uniquely
determine the configuration for a double pendulum -- it can wander the
constant-energy shell without ever landing both links upright. And
warm-starting the long-horizon MPC with the heuristic's own trajectory
just drops IPOPT (a local optimizer) into whatever "valley" of the
non-convex landscape that trajectory sits in -- it won't jump valleys.

Fix: hard-constrain both endpoints (hanging -> upright, exactly, both at
rest) and minimize control effort only, then seed IPOPT with a naive
straight-line interpolation between the two states in state space. That
guess ignores dynamics completely (not a valid trajectory at all) but
starts IPOPT in the *topologically correct* valley -- the one that
actually connects bottom to top -- and IPOPT bends it into a dynamically
consistent path from there.

Track the resulting reference with a short-horizon MPC (mpc.py) for
robustness to model mismatch/disturbance, rather than replaying the
open-loop control sequence blind.
"""
import numpy as np
import casadi as ca

from mpc import _dynamics, _rk4_step, MOTOR_FORCE_MAX, MOTOR_FREE_SPEED, NX


def solve_swingup_trajectory(params, x0, xf, Np=150, dt=0.05, s_max=0.18,
                              max_iter=3000):
    """
    Direct collocation BVP: find X(0..Np), U(0..Np-1) such that dynamics
    are satisfied at every shooting node, X[:,0]=x0, X[:,Np]=xf exactly,
    subject to the same rail/motor-derating constraints as the MPC, and
    minimizing total control effort sum(u_k^2).

    Returns (X_opt, U_opt, success). X_opt is (6, Np+1), U_opt is (Np,).
    """
    total_mass = params['M'] + params['m1'] + params['m2']
    sddot_max = MOTOR_FORCE_MAX / total_mass

    opti = ca.Opti()
    X = opti.variable(NX, Np + 1)
    U = opti.variable(1, Np)

    J = 0
    for k in range(Np):
        J += U[0, k] ** 2
        opti.subject_to(X[:, k + 1] == _rk4_step(X[:, k], U[0, k], params, dt))

        xdot_k = _dynamics(X[:, k], U[0, k], params)
        opti.subject_to(opti.bounded(-sddot_max, xdot_k[3], sddot_max))

        u_avail_k = MOTOR_FORCE_MAX * (1 - (X[3, k] / MOTOR_FREE_SPEED) ** 2)
        opti.subject_to(opti.bounded(-u_avail_k, U[0, k], u_avail_k))
    opti.minimize(J)

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, Np] == xf)
    opti.subject_to(opti.bounded(-s_max, X[0, :], s_max))
    opti.subject_to(opti.bounded(-MOTOR_FREE_SPEED, X[3, :], MOTOR_FREE_SPEED))

    # naive straight-line kinematic warm start -- ignores dynamics
    # entirely, but starts the solver in the valley that actually
    # connects x0 to xf instead of wherever a "physically simulated"
    # guess happens to sit.
    X_guess = np.linspace(x0, xf, Np + 1).T
    opti.set_initial(X, X_guess)
    opti.set_initial(U, np.zeros((1, Np)))

    opti.solver('ipopt', {'print_time': 0},
                {'print_level': 0, 'sb': 'yes', 'max_iter': max_iter})
    try:
        sol = opti.solve()
        return sol.value(X), sol.value(U), True
    except RuntimeError:
        return opti.debug.value(X), opti.debug.value(U), False


def _linearize_fns(params, dt):
    x = ca.MX.sym('x', NX)
    u = ca.MX.sym('u')
    x_next = _rk4_step(x, u, params, dt)
    A_fun = ca.Function('A', [x, u], [ca.jacobian(x_next, x)])
    B_fun = ca.Function('B', [x, u], [ca.jacobian(x_next, u)])
    return A_fun, B_fun


def tvlqr_gains(params, X_ref, U_ref, dt, Q, R, Qf):
    """
    Backward Riccati recursion for time-varying LQR feedback gains
    around the reference trajectory. Necessary because this trajectory
    passes through the unstable inverted equilibrium -- confirmed by
    testing: replaying U_ref open-loop diverges (final th2 off by ~13
    rad) since tiny numerical mismatch between the CasADi and torch RK4
    implementations gets chaotically amplified with no feedback to
    correct it.
    """
    A_fun, B_fun = _linearize_fns(params, dt)
    Np = U_ref.shape[0]
    Ks = [None] * Np
    S = Qf.copy()
    for k in reversed(range(Np)):
        A = np.array(A_fun(X_ref[:, k], U_ref[k]))
        B = np.array(B_fun(X_ref[:, k], U_ref[k])).reshape(NX, 1)
        S_BB = R + B.T @ S @ B
        K = np.linalg.solve(S_BB, B.T @ S @ A)  # (1,6)
        Ks[k] = K
        S = Q + A.T @ S @ A - A.T @ S @ B @ K
    return Ks


def track_trajectory(params, X_ref, U_ref, dt, s_max=0.18,
                      Q=None, R=None, Qf=None):
    """
    Closed-loop: step the *actual* plant (dynamics.py) forward, applying
    u_k = u_ref[k] - K_k @ (x_k - x_ref[k]) with time-varying LQR gains
    computed around the reference. Feedforward (U_ref) does the bulk of
    the work; the TVLQR term corrects small deviations before they can
    compound through the unstable region near upright.
    """
    import torch
    from dynamics import forward_dynamics

    if Q is None:
        Q = np.diag([10.0, 50.0, 50.0, 1.0, 5.0, 5.0])
    if R is None:
        R = np.array([[0.01]])
    if Qf is None:
        Qf = 10 * Q

    Ks = tvlqr_gains(params, X_ref, U_ref, dt, Q, R, Qf)

    def rk4(state, u):
        st = torch.tensor(state, dtype=torch.float64)
        ut = torch.tensor(float(u), dtype=torch.float64)
        k1 = forward_dynamics(st, ut, params).numpy()
        k2 = forward_dynamics(torch.tensor(state + dt / 2 * k1), ut, params).numpy()
        k3 = forward_dynamics(torch.tensor(state + dt / 2 * k2), ut, params).numpy()
        k4 = forward_dynamics(torch.tensor(state + dt * k3), ut, params).numpy()
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    Np_ref = U_ref.shape[0]
    state = X_ref[:, 0].copy()
    log = [state.copy()]

    for k in range(Np_ref):
        dx = state - X_ref[:, k]
        u = U_ref[k] - float(Ks[k] @ dx)
        u = np.clip(u, -MOTOR_FORCE_MAX, MOTOR_FORCE_MAX)
        state = rk4(state, u)
        log.append(state.copy())

    return np.array(log).T  # (6, Np_ref+1)


if __name__ == "__main__":
    params = dict(
        m1=0.2, m2=0.15, l1=0.3, l2=0.25, M=1.0,
        I1=0.2 * 0.3 ** 2 / 12, I2=0.15 * 0.25 ** 2 / 12,
    )
    x0 = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])  # both links hanging down
    xf = np.zeros(6)
    dt = 0.05

    print("solving offline swing-up trajectory (direct collocation)...")
    X_ref, U_ref, ok = solve_swingup_trajectory(params, x0, xf, Np=150, dt=dt, s_max=0.18)
    print(f"solve success: {ok}")
    print(f"endpoint reached: {X_ref[:, -1]}")
    print(f"max|s|={np.max(np.abs(X_ref[0])):.4f}  max|u|={np.max(np.abs(U_ref)):.3f}")

    print("\ntracking reference in closed loop against dynamics.py plant...")
    X_track = track_trajectory(params, X_ref, U_ref, dt, s_max=0.18)
    final_err = np.abs(X_track[:, -1] - xf)
    print(f"closed-loop final state: {X_track[:, -1]}")
    print(f"final tracking error vs upright target: {final_err}")

    print(f"\n{'t':>5} {'s_ref':>8} {'s_track':>8} {'th1_ref':>8} {'th1_track':>9}")
    for k in range(0, X_ref.shape[1], 15):
        print(f"{k*dt:5.2f} {X_ref[0,k]:8.4f} {X_track[0,k]:8.4f} "
              f"{X_ref[1,k]:8.4f} {X_track[1,k]:9.4f}")
