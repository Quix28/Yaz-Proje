"""
Forward dynamics for cart + double inverted pendulum.

State:   x = [s, th1, th2, sdot, th1dot, th2dot]
Control: u  -- net force applied to cart [N]. Voltage->force mapping
              via motor constants is a separate actuator model, added
              after hardware system-ID (Step 7).
Params (dict, values broadcastable to state.shape[:-1]):
    m1, m2   -- link masses [kg]
    l1, l2   -- link lengths, pivot-to-pivot [m]
    I1, I2   -- link moments of inertia about own COM [kg m^2]
                (uniform rod default: I_i = m_i*l_i^2/12)
    M        -- cart mass [kg]
    lc1, lc2 -- pivot-to-COM distance [m] (default l_i/2, uniform rod)
    g        -- gravity [m/s^2] (default 9.81)
    b0, b1, b2 -- viscous damping: cart, joint1, joint2 (default 0)

theta measured from vertical-up.
"""
import torch


def forward_dynamics(state, u, params):
    """
    state: (..., 6) tensor [s, th1, th2, sdot, th1dot, th2dot]
    u:     (...,) tensor, force on cart [N]
    params: dict, keys m1, m2, l1, l2, I1, I2, M, plus optional
            g, b0, b1, b2, lc1, lc2

    returns: (..., 6) tensor, xdot
    """
    s, th1, th2, sdot, th1dot, th2dot = torch.unbind(state, dim=-1)

    m1, m2 = params["m1"], params["m2"]
    l1, l2 = params["l1"], params["l2"]
    I1, I2 = params["I1"], params["I2"]
    M = params["M"]
    g = params.get("g", 9.81)
    b0 = params.get("b0", 0.0)
    b1 = params.get("b1", 0.0)
    b2 = params.get("b2", 0.0)
    lc1 = params.get("lc1", l1 / 2)
    lc2 = params.get("lc2", l2 / 2)

    c1, s1 = torch.cos(th1), torch.sin(th1)
    c2, s2 = torch.cos(th2), torch.sin(th2)
    c12 = torch.cos(th1 - th2)
    s12 = torch.sin(th1 - th2)

    ones = torch.ones_like(s)
    M11 = (M + m1 + m2) * ones
    M12 = (m1 * lc1 + m2 * l1) * c1
    M13 = m2 * lc2 * c2
    M22 = (m1 * lc1**2 + m2 * l1**2 + I1) * ones
    M23 = m2 * l1 * lc2 * c12
    M33 = (m2 * lc2**2 + I2) * ones

    Mmat = torch.stack([
        torch.stack([M11, M12, M13], dim=-1),
        torch.stack([M12, M22, M23], dim=-1),
        torch.stack([M13, M23, M33], dim=-1),
    ], dim=-2)

    rhs_s = (u - b0 * sdot
              + (m1 * lc1 + m2 * l1) * s1 * th1dot**2
              + m2 * lc2 * s2 * th2dot**2)

    rhs_th1 = (-b1 * th1dot
               - m2 * l1 * lc2 * s12 * th2dot**2
               + (m1 * lc1 + m2 * l1) * g * s1)

    rhs_th2 = (-b2 * th2dot
               + m2 * l1 * lc2 * s12 * th1dot**2
               + m2 * lc2 * g * s2)

    rhs = torch.stack([rhs_s, rhs_th1, rhs_th2], dim=-1)

    qddot = torch.linalg.solve(Mmat, rhs.unsqueeze(-1)).squeeze(-1)
    sddot, th1ddot, th2ddot = torch.unbind(qddot, dim=-1)

    return torch.stack([sdot, th1dot, th2dot, sddot, th1ddot, th2ddot], dim=-1)
