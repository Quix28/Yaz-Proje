"""
Symbolic derivation of cart + double inverted pendulum dynamics
via Euler-Lagrange. Run to print the equations of motion, cross-check
against the closed-form implementation in dynamics.py.

Deliberately a skeleton, not the full model: encoder masses (m_enc1,
m_enc2) and Coulomb friction (dynamics.py's tanh(qdot/eps) term) are
omitted here since they'd clutter the symbolic derivation without
changing the structure being checked (Lagrangian -> EOM via
Euler-Lagrange). Cross-check the *shape* of the equations here; treat
dynamics.py as the source of truth for the full model.
"""
import sympy as sp

t = sp.symbols('t')
s_f, th1_f, th2_f = sp.symbols('s theta1 theta2', cls=sp.Function)
s, th1, th2 = s_f(t), th1_f(t), th2_f(t)

m1, m2, M, l1, l2, lc1, lc2, I1, I2, g, u, b0, b1, b2 = sp.symbols(
    'm1 m2 M l1 l2 lc1 lc2 I1 I2 g u b0 b1 b2', positive=True
)

sdot, th1dot, th2dot = s.diff(t), th1.diff(t), th2.diff(t)

# positions
x1 = s + lc1 * sp.sin(th1)
y1 = lc1 * sp.cos(th1)
x2 = s + l1 * sp.sin(th1) + lc2 * sp.sin(th2)
y2 = l1 * sp.cos(th1) + lc2 * sp.cos(th2)

x1d, y1d = x1.diff(t), y1.diff(t)
x2d, y2d = x2.diff(t), y2.diff(t)

T = (sp.Rational(1, 2) * M * sdot**2
     + sp.Rational(1, 2) * m1 * (x1d**2 + y1d**2) + sp.Rational(1, 2) * I1 * th1dot**2
     + sp.Rational(1, 2) * m2 * (x2d**2 + y2d**2) + sp.Rational(1, 2) * I2 * th2dot**2)

V = m1 * g * y1 + m2 * g * y2

L = sp.simplify(T - V)

Q = {s: u - b0 * sdot, th1: -b1 * th1dot, th2: -b2 * th2dot}

for name, q in zip(['s', 'theta1', 'theta2'], (s, th1, th2)):
    qd = q.diff(t)
    eq = sp.diff(L, qd).diff(t) - sp.diff(L, q) - Q[q]
    eq = sp.simplify(eq)
    print(f"\n-- {name} equation (=0) --")
    sp.pprint(eq)
