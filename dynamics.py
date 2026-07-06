"""
Forward dynamics for cart + double inverted pendulum.

State:   x = [s, th1, th2, sdot, th1dot, th2dot]
Control: u  -- net force applied to cart [N]. Voltage->force mapping
              via motor constants is a separate actuator model, added
              after hardware system-ID (Step 7).
Params (dict, values broadcastable to state.shape[:-1]):
    m1, m2   -- link (rod only) masses [kg] -- do NOT lump encoder mass
                in here; m_enc1/m_enc2 below track those separately
                since they don't share the rod's uniform-mass-distribution
                assumption (I_i = m_i*l_i^2/12 is only valid for the bare
                rod, not a rod-plus-concentrated-point-mass system)
    l1, l2   -- link lengths, pivot-to-pivot [m]
    I1, I2   -- link (rod only) moments of inertia about own COM [kg m^2]
                (uniform rod default: I_i = m_i*l_i^2/12)
    M        -- cart mass [kg], NOT including encoder1 (see m_enc1)
    m_enc1   -- mass of the theta1 optical encoder [kg], mounted on the
                cart itself (pure translating mass, no rotational
                coupling -- folds directly into effective cart mass)
    m_enc2   -- mass of the theta2 optical encoder [kg], mounted rigidly
                at the link1/link2 joint -- rotates with theta1 only, has
                no theta2 dependence (contributes to the l1-arm terms:
                M12, M22's l1^2 term, and theta1's gravity/centrifugal
                terms; leaves theta2's equation and all lc2-terms
                untouched). Verified via sympy Euler-Lagrange, not just
                pattern-matched.
    lc1, lc2 -- pivot-to-COM distance [m] (default l_i/2, uniform rod)
    g        -- gravity [m/s^2] (default 9.81)
    b0, b1, b2   -- viscous damping: cart, joint1, joint2 [N*s/m, N*m*s/rad]
                    default small nonzero (lubricated bearing + encoder
                    drag), not 0 -- override to 0 for an idealized/
                    frictionless model.
    cf0, cf1, cf2 -- Coulomb (dry) friction magnitude: cart rail,
                    joint1, joint2 [N, N*m, N*m]. Defaults from typical
                    small ball-bearing + 600 P/R optical encoder specs
                    (bearing ~1-2 mN*m starting torque, encoder ~1 mN*m),
                    not measured on your actual rig -- replace after
                    system-ID (CLAUDE.md Step 7).

theta measured from vertical-up.
"""
import torch

# smoothing width for the tanh(qdot/eps) approximation to sign(qdot) in
# the Coulomb friction term -- real dry friction is discontinuous at
# zero velocity, which breaks gradients/solvers; this keeps it smooth
# while still closely approximating a hard sign() away from qdot=0.
# 0.05 (not tighter): a sharper kink here made the swing-up NLP too
# stiff for IPOPT (the trajectory crosses zero velocity many times while
# pumping, and each crossing hit the near-vertical tanh edge, exhausting
# the iteration budget before convergence). mpc.py uses the same value
# so the solver's model stays identical to this plant.
FRICTION_EPS = 0.05


def forward_dynamics(state, u, params):
    """
    state: (..., 6) tensor [s, th1, th2, sdot, th1dot, th2dot]
    u:     (...,) tensor, force on cart [N]
    params: dict, keys m1, m2, l1, l2, I1, I2, M, plus optional
            g, b0, b1, b2, cf0, cf1, cf2, m_enc1, m_enc2, lc1, lc2

    returns: (..., 6) tensor, xdot
    """
    s, th1, th2, sdot, th1dot, th2dot = torch.unbind(state, dim=-1)

    m1, m2 = params["m1"], params["m2"]
    l1, l2 = params["l1"], params["l2"]
    I1, I2 = params["I1"], params["I2"]
    M = params["M"]
    g = params.get("g", 9.81)
    b0 = params.get("b0", 0.02)
    b1 = params.get("b1", 0.0008)
    b2 = params.get("b2", 0.0008)
    cf0 = params.get("cf0", 0.05)
    cf1 = params.get("cf1", 0.0025)
    cf2 = params.get("cf2", 0.0025)
    m_enc1 = params.get("m_enc1", 0.18)
    m_enc2 = params.get("m_enc2", 0.18)
    lc1 = params.get("lc1", l1 / 2)
    lc2 = params.get("lc2", l2 / 2)

    fric_s = b0 * sdot + cf0 * torch.tanh(sdot / FRICTION_EPS)
    fric_th1 = b1 * th1dot + cf1 * torch.tanh(th1dot / FRICTION_EPS)
    fric_th2 = b2 * th2dot + cf2 * torch.tanh(th2dot / FRICTION_EPS)

    c1, s1 = torch.cos(th1), torch.sin(th1)
    c2, s2 = torch.cos(th2), torch.sin(th2)
    c12 = torch.cos(th1 - th2)
    s12 = torch.sin(th1 - th2)

    M_eff = M + m_enc1       # encoder1 rides on the cart, pure translation
    m2_l1 = m2 + m_enc2      # encoder2 rides at the joint, rotates with th1 only

    ones = torch.ones_like(s)
    M11 = (M_eff + m1 + m2_l1) * ones
    M12 = (m1 * lc1 + m2_l1 * l1) * c1
    M13 = m2 * lc2 * c2
    M22 = (m1 * lc1**2 + m2_l1 * l1**2 + I1) * ones
    M23 = m2 * l1 * lc2 * c12
    M33 = (m2 * lc2**2 + I2) * ones

    Mmat = torch.stack([
        torch.stack([M11, M12, M13], dim=-1),
        torch.stack([M12, M22, M23], dim=-1),
        torch.stack([M13, M23, M33], dim=-1),
    ], dim=-2)

    rhs_s = (u - fric_s
              + (m1 * lc1 + m2_l1 * l1) * s1 * th1dot**2
              + m2 * lc2 * s2 * th2dot**2)

    rhs_th1 = (-fric_th1
               - m2 * l1 * lc2 * s12 * th2dot**2
               + (m1 * lc1 + m2_l1 * l1) * g * s1)

    rhs_th2 = (-fric_th2
               + m2 * l1 * lc2 * s12 * th1dot**2
               + m2 * lc2 * g * s2)

    rhs = torch.stack([rhs_s, rhs_th1, rhs_th2], dim=-1)

    qddot = torch.linalg.solve(Mmat, rhs.unsqueeze(-1)).squeeze(-1)
    sddot, th1ddot, th2ddot = torch.unbind(qddot, dim=-1)

    return torch.stack([sdot, th1dot, th2dot, sddot, th1ddot, th2ddot], dim=-1)
