"""
MPC teacher for cart + double inverted pendulum. CasADi + IPOPT,
multiple shooting, RK4 discretization. Dynamics ported from dynamics.py
(same mass matrix / rhs terms, CasADi symbolics instead of torch).

Control u: net force on cart [N], not motor voltage -- voltage->force
mapping deferred to actuator model post system-ID (CLAUDE.md Step 7).
u_max=12 here is a placeholder bound carried over from the 12V spec;
swap in real force limit once motor constants (Kt, Kb, R, r) are known.
"""
import casadi as ca
import numpy as np

NX = 6  # s, th1, th2, sdot, th1dot, th2dot
NU = 1


def _dynamics(x, u, p):
    s, th1, th2, sdot, th1dot, th2dot = ca.vertsplit(x)

    m1, m2, l1, l2 = p['m1'], p['m2'], p['l1'], p['l2']
    I1, I2, M = p['I1'], p['I2'], p['M']
    g = p.get('g', 9.81)
    b0, b1, b2 = p.get('b0', 0.0), p.get('b1', 0.0), p.get('b2', 0.0)
    lc1 = p.get('lc1', l1 / 2)
    lc2 = p.get('lc2', l2 / 2)

    c1, s1 = ca.cos(th1), ca.sin(th1)
    c2, s2 = ca.cos(th2), ca.sin(th2)
    c12 = ca.cos(th1 - th2)
    s12 = ca.sin(th1 - th2)

    M11 = M + m1 + m2
    M12 = (m1 * lc1 + m2 * l1) * c1
    M13 = m2 * lc2 * c2
    M22 = m1 * lc1**2 + m2 * l1**2 + I1
    M23 = m2 * l1 * lc2 * c12
    M33 = m2 * lc2**2 + I2

    Mmat = ca.vertcat(
        ca.horzcat(M11, M12, M13),
        ca.horzcat(M12, M22, M23),
        ca.horzcat(M13, M23, M33),
    )

    rhs_s = (u - b0 * sdot
             + (m1 * lc1 + m2 * l1) * s1 * th1dot**2
             + m2 * lc2 * s2 * th2dot**2)
    rhs_th1 = (-b1 * th1dot
               - m2 * l1 * lc2 * s12 * th2dot**2
               + (m1 * lc1 + m2 * l1) * g * s1)
    rhs_th2 = (-b2 * th2dot
               + m2 * l1 * lc2 * s12 * th1dot**2
               + m2 * lc2 * g * s2)

    rhs = ca.vertcat(rhs_s, rhs_th1, rhs_th2)
    qddot = ca.solve(Mmat, rhs)

    return ca.vertcat(sdot, th1dot, th2dot, qddot[0], qddot[1], qddot[2])


def _rk4_step(x, u, p, dt):
    k1 = _dynamics(x, u, p)
    k2 = _dynamics(x + dt / 2 * k1, u, p)
    k3 = _dynamics(x + dt / 2 * k2, u, p)
    k4 = _dynamics(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class MPCController:
    """
    Finite-horizon NMPC, upright regulation via multiple shooting.
    Builds NLP once; call .solve(x0) each control step (warm-started
    from previous solution, shifted by one step).
    """

    def __init__(self, params, Np=20, dt=0.05,
                 Q=None, R=1e-2, Qf=None,
                 s_max=0.23, u_max=12.0,
                 sdot_max=None, thdot_max=None):
        self.Np = Np
        self.dt = dt
        self.params = params
        self.s_max = s_max
        self.u_max = u_max

        if Q is None:
            Q = np.diag([50.0, 200.0, 200.0, 1.0, 5.0, 5.0])
        if Qf is None:
            Qf = 10 * Q
        self.Q, self.Qf, self.R = Q, Qf, R

        opti = ca.Opti()
        X = opti.variable(NX, Np + 1)
        U = opti.variable(NU, Np)
        x0_param = opti.parameter(NX)
        xref_param = opti.parameter(NX)

        J = 0
        for k in range(Np):
            dx = X[:, k] - xref_param
            J += ca.mtimes([dx.T, Q, dx]) + R * U[0, k]**2
            opti.subject_to(X[:, k + 1] == _rk4_step(X[:, k], U[0, k], params, dt))

        dxN = X[:, Np] - xref_param
        J += ca.mtimes([dxN.T, Qf, dxN])
        opti.minimize(J)

        opti.subject_to(X[:, 0] == x0_param)
        opti.subject_to(opti.bounded(-s_max, X[0, :], s_max))
        opti.subject_to(opti.bounded(-u_max, U[0, :], u_max))
        if sdot_max is not None:
            opti.subject_to(opti.bounded(-sdot_max, X[3, :], sdot_max))
        if thdot_max is not None:
            opti.subject_to(opti.bounded(-thdot_max, X[4, :], thdot_max))
            opti.subject_to(opti.bounded(-thdot_max, X[5, :], thdot_max))

        opti.solver('ipopt', {'print_time': 0}, {'print_level': 0, 'sb': 'yes'})

        self.opti = opti
        self.X, self.U = X, U
        self.x0_param, self.xref_param = x0_param, xref_param
        self._X_prev = np.zeros((NX, Np + 1))
        self._U_prev = np.zeros((1, Np))

    def solve(self, x0, xref=None):
        """
        x0:   (6,) current state
        xref: (6,) target state, default upright/centered (zeros)
        returns: (u0, X_pred, U_pred)
        """
        if xref is None:
            xref = np.zeros(NX)

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.xref_param, xref)
        self.opti.set_initial(self.X, self._X_prev)
        self.opti.set_initial(self.U, self._U_prev)

        sol = self.opti.solve()

        X_sol = sol.value(self.X)
        U_sol = np.atleast_2d(sol.value(self.U))

        self._X_prev = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        self._U_prev = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

        return float(U_sol[0, 0]), X_sol, U_sol


if __name__ == "__main__":
    params = dict(
        m1=0.2, m2=0.15, l1=0.3, l2=0.25, M=1.0,
        I1=0.2 * 0.3**2 / 12, I2=0.15 * 0.25**2 / 12,
    )
    ctrl = MPCController(params, Np=20, dt=0.05, s_max=0.23, u_max=12.0)

    # both links hanging straight down (stable equilibrium under gravity) --
    # swing-up task, not small-angle stabilization
    x = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    print(f"{'step':>4} {'s':>8} {'th1':>8} {'th2':>8} {'u':>8}")
    for i in range(120):
        u, X_pred, _ = ctrl.solve(x)
        x = X_pred[:, 1]
        print(f"{i:4d} {x[0]:8.4f} {x[1]:8.4f} {x[2]:8.4f} {u:8.3f}")
