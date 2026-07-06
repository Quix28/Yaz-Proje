"""
MPC teacher for cart + double inverted pendulum. CasADi + IPOPT,
multiple shooting, RK4 discretization. Dynamics ported from dynamics.py
(same mass matrix / rhs terms, CasADi symbolics instead of torch).

Control u: net force on cart [N], not motor voltage -- full electrical
actuator model (Kt, Kb, R) still deferred to post system-ID (CLAUDE.md
Step 7). MOTOR_FORCE_MAX below is a catalog-spec estimate, not measured
on your actual unit -- replace once you have real numbers.
"""
import casadi as ca
import numpy as np

NX = 6  # s, th1, th2, sdot, th1dot, th2dot
NU = 1

# NEMA23 stepper, GT2 belt direct-drive (20-tooth pulley, no gearbox) --
# upgraded from NEMA17 since that motor's practical spec ceiling (~2x
# baseline mass, 1.5x baseline length before running out of force
# margin) was reached. Adjust constants if your actual unit differs
# (different pulley, geared motor, lead screw, etc).
NEMA23_HOLDING_TORQUE = 1.9    # N*m, typical mid-range NEMA23 (e.g. 3A
                               # 23HS30-series) -- catalog spec, not
                               # measured; higher-current variants go to
                               # ~2.8-3.0 N*m if you need more headroom
GT2_PULLEY_RADIUS = 0.006366   # m, 20-tooth GT2 pulley pitch radius (unchanged)
DYNAMIC_DERATE = 0.5           # holding-torque -> usable dynamic torque;
                               # steppers lose torque with speed (back-EMF)
                               # and skip steps if driven at rated holding
                               # torque continuously -- 50% margin against that
MOTOR_FORCE_MAX = NEMA23_HOLDING_TORQUE * DYNAMIC_DERATE / GT2_PULLEY_RADIUS  # ~149 N

# Steppers have essentially zero holding torque left well before their
# absolute max RPM -- 600 RPM is a rough ceiling for *reliable, in-torque*
# operation with a common driver (A4988/DRV8825 class), not the motor's
# theoretical no-load top speed. Kept at the same value as the NEMA17
# case as a placeholder; NEMA23 frames often have higher inductance and
# may actually top out at a *lower* reliable RPM than NEMA17 on the same
# driver/voltage -- verify against your actual driver's current/voltage
# once picked. Translate through the same pulley to get a linear
# cart-speed ceiling, and derate available force to zero as the cart
# approaches it (crude linear torque-speed curve).
MOTOR_MAX_RPM = 600
MOTOR_FREE_SPEED = MOTOR_MAX_RPM * 2 * np.pi / 60 * GT2_PULLEY_RADIUS  # ~0.4 m/s

# same friction model as dynamics.py (viscous + smoothed Coulomb from
# typical bearing/optical-encoder specs) -- kept in sync so the solver's
# internal model matches the actual plant. See dynamics.py for why 0.05
# rather than a sharper value.
FRICTION_EPS = 0.05


def _dynamics(x, u, p):
    s, th1, th2, sdot, th1dot, th2dot = ca.vertsplit(x)

    m1, m2, l1, l2 = p['m1'], p['m2'], p['l1'], p['l2']
    I1, I2, M = p['I1'], p['I2'], p['M']
    g = p.get('g', 9.81)
    b0, b1, b2 = p.get('b0', 0.02), p.get('b1', 0.0008), p.get('b2', 0.0008)
    cf0, cf1, cf2 = p.get('cf0', 0.05), p.get('cf1', 0.0025), p.get('cf2', 0.0025)
    m_enc1 = p.get('m_enc1', 0.18)  # theta1 encoder, on the cart
    m_enc2 = p.get('m_enc2', 0.18)  # theta2 encoder, at the joint (rotates with th1 only)
    lc1 = p.get('lc1', l1 / 2)
    lc2 = p.get('lc2', l2 / 2)

    c1, s1 = ca.cos(th1), ca.sin(th1)
    c2, s2 = ca.cos(th2), ca.sin(th2)
    c12 = ca.cos(th1 - th2)
    s12 = ca.sin(th1 - th2)

    M_eff = M + m_enc1
    m2_l1 = m2 + m_enc2

    M11 = M_eff + m1 + m2_l1
    M12 = (m1 * lc1 + m2_l1 * l1) * c1
    M13 = m2 * lc2 * c2
    M22 = m1 * lc1**2 + m2_l1 * l1**2 + I1
    M23 = m2 * l1 * lc2 * c12
    M33 = m2 * lc2**2 + I2

    Mmat = ca.vertcat(
        ca.horzcat(M11, M12, M13),
        ca.horzcat(M12, M22, M23),
        ca.horzcat(M13, M23, M33),
    )

    fric_s = b0 * sdot + cf0 * ca.tanh(sdot / FRICTION_EPS)
    fric_th1 = b1 * th1dot + cf1 * ca.tanh(th1dot / FRICTION_EPS)
    fric_th2 = b2 * th2dot + cf2 * ca.tanh(th2dot / FRICTION_EPS)

    rhs_s = (u - fric_s
             + (m1 * lc1 + m2_l1 * l1) * s1 * th1dot**2
             + m2 * lc2 * s2 * th2dot**2)
    rhs_th1 = (-fric_th1
               - m2 * l1 * lc2 * s12 * th2dot**2
               + (m1 * lc1 + m2_l1 * l1) * g * s1)
    rhs_th2 = (-fric_th2
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
                 s_max=0.18, u_max=MOTOR_FORCE_MAX,
                 sdot_max=None, thdot_max=None, sddot_max=None):
        # s_max default assumes 50cm MGN12 rail (physical +-0.25m from
        # center) minus ~8cm cart footprint minus 3cm safety buffer per
        # side -- tighten/loosen once actual carriage width is measured.
        #
        # sddot_max default: derived from MOTOR_FORCE_MAX and this
        # instance's own total moving mass (M + m1 + m2), i.e. the max
        # acceleration the motor could impart if it were the only force
        # acting. Enforced separately from u_max below because pendulum
        # coupling terms (centrifugal/gravity reaction through the links)
        # can push actual cart acceleration above force/mass alone.
        #
        # sdot_max default: MOTOR_FREE_SPEED -- a stepper's usable torque
        # collapses well before its absolute top RPM, so cart speed is
        # capped there and available force is derated to zero as it's
        # approached (see u_avail_k below), rather than staying flat at
        # u_max regardless of how fast the cart is already moving.
        self.Np = Np
        self.dt = dt
        self.params = params
        self.s_max = s_max
        self.u_max = u_max
        if sddot_max is None:
            total_mass = params['M'] + params['m1'] + params['m2']
            sddot_max = MOTOR_FORCE_MAX / total_mass
        self.sddot_max = sddot_max
        if sdot_max is None:
            sdot_max = MOTOR_FREE_SPEED
        self.sdot_max = sdot_max

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

            xdot_k = _dynamics(X[:, k], U[0, k], params)
            opti.subject_to(opti.bounded(-sddot_max, xdot_k[3], sddot_max))

            # torque-speed derate: force available shrinks to 0 as cart
            # speed approaches sdot_max (crude linear stepper torque curve)
            u_avail_k = u_max * (1 - (X[3, k] / sdot_max) ** 2)
            opti.subject_to(opti.bounded(-u_avail_k, U[0, k], u_avail_k))

        dxN = X[:, Np] - xref_param
        J += ca.mtimes([dxN.T, Qf, dxN])
        opti.minimize(J)

        opti.subject_to(X[:, 0] == x0_param)
        opti.subject_to(opti.bounded(-s_max, X[0, :], s_max))
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
        # m1, m2 are rod-only mass -- encoder mass (180g each) tracked
        # separately via m_enc1 (on the cart) / m_enc2 (at the joint,
        # rotates with th1 only) since it doesn't share the rod's
        # uniform-mass-distribution assumption used for I1, I2.
        m1=0.12, m2=0.09, l1=0.3, l2=0.25, M=1.0,
        m_enc1=0.18, m_enc2=0.18,
        I1=0.12 * 0.3**2 / 12, I2=0.09 * 0.25**2 / 12,
    )
    ctrl = MPCController(params, Np=20, dt=0.05, s_max=0.18)

    # both links hanging straight down (stable equilibrium under gravity) --
    # swing-up task, not small-angle stabilization
    x = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    print(f"{'step':>4} {'s':>8} {'th1':>8} {'th2':>8} {'u':>8}")
    for i in range(120):
        u, X_pred, _ = ctrl.solve(x)
        x = X_pred[:, 1]
        print(f"{i:4d} {x[0]:8.4f} {x[1]:8.4f} {x[2]:8.4f} {u:8.3f}")
