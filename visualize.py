"""
Animate a swing-up run -- cart + double pendulum on the rail, viewed
from the side. theta measured from vertical-up, so link tip = pivot +
l*(sin(theta), cos(theta)).

Uses trajectory.py (offline direct-collocation BVP + TVLQR tracking),
the swing-up approach that actually converges to upright -- swingup.py's
energy-shaping heuristic plateaus short of it (see trajectory.py's
docstring for why).

Run directly for a live matplotlib window. Pass --save path.gif to also
write an animated gif (works headless, no display needed).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from trajectory import solve_swingup_trajectory, track_trajectory

# cart footprint assumed ~8cm (MGN12 carriage + motor mount) -- adjust
# once actual carriage width is measured
CART_W, CART_H = 0.08, 0.05


def animate(log, params, s_max=0.18, track_max=0.25, dt=0.05, save_path=None, fps=None):
    l1, l2 = params["l1"], params["l2"]

    S = np.array([row[1] for row in log])
    TH1 = np.array([row[2] for row in log])
    TH2 = np.array([row[3] for row in log])
    U = np.array([row[7] for row in log])
    steps = len(log)

    reach = l1 + l2 + 0.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-track_max - 0.1, track_max + 0.1)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax.axhline(0, color="k", lw=1)
    ax.axvline(-s_max, color="r", ls="--", lw=1, alpha=0.6)
    ax.axvline(s_max, color="r", ls="--", lw=1, alpha=0.6)
    ax.axvline(-track_max, color="gray", lw=2)
    ax.axvline(track_max, color="gray", lw=2)

    cart = Rectangle((-CART_W / 2, -CART_H / 2), CART_W, CART_H,
                      fc="steelblue", ec="k", zorder=3)
    ax.add_patch(cart)

    link1_line, = ax.plot([], [], "o-", lw=3, color="darkorange", zorder=4)
    link2_line, = ax.plot([], [], "o-", lw=3, color="crimson", zorder=4)
    info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         va="top", family="monospace")

    def joints(i):
        s, th1, th2 = S[i], TH1[i], TH2[i]
        p0 = (s, 0.0)
        p1 = (s + l1 * np.sin(th1), l1 * np.cos(th1))
        p2 = (p1[0] + l2 * np.sin(th2), p1[1] + l2 * np.cos(th2))
        return p0, p1, p2

    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        info_text.set_text("")
        return cart, link1_line, link2_line, info_text

    def update(i):
        p0, p1, p2 = joints(i)
        cart.set_xy((p0[0] - CART_W / 2, -CART_H / 2))
        link1_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        mode_str = f" [{log[i][8]}]" if len(log[i]) > 8 else ""
        t_val = log[i][0] if len(log[i]) > 0 else i * dt
        
        info_text.set_text(
            f"t={t_val:5.2f}s  s={S[i]:+.3f}m  th1={TH1[i]:+.2f}  "
            f"th2={TH2[i]:+.2f}  u={U[i]:+.1f}{mode_str}"
        )
        return cart, link1_line, link2_line, info_text

    anim = animation.FuncAnimation(
        fig, update, frames=steps, init_func=init,
        interval=dt * 1000, blit=True,
    )

    if save_path:
        anim.save(save_path, writer=animation.PillowWriter(fps=fps or round(1 / dt)))
        print(f"saved {save_path}")

    return fig, anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None,
                         help="path to write an animated gif")
    parser.add_argument("--no-show", action="store_true",
                         help="skip opening a live window (e.g. headless run)")
    args = parser.parse_args()

    params = dict(
        # m1, m2 are rod-only mass -- encoder mass (180g each) tracked
        # separately via m_enc1 (on the cart) / m_enc2 (at the joint,
        # rotates with th1 only)
        m1=0.12, m2=0.09, l1=0.3, l2=0.25, M=1.0,
        m_enc1=0.18, m_enc2=0.18,
        I1=0.12 * 0.3 ** 2 / 12, I2=0.09 * 0.25 ** 2 / 12,
    )
    x0 = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])  # both links hanging down
    xf = np.zeros(6)
    dt = 0.05

    print("solving offline swing-up trajectory (direct collocation)...")
    X_ref, U_ref, ok = solve_swingup_trajectory(params, x0, xf, Np=150, dt=dt, s_max=0.18)
    print(f"solve success: {ok}")

    print("tracking with TVLQR against the dynamics.py plant...")
    X_track = track_trajectory(params, X_ref, U_ref, dt, s_max=0.18)

    log = []
    for k in range(X_track.shape[1]):
        s, th1, th2, sdot, th1dot, th2dot = X_track[:, k]
        u = U_ref[k] if k < len(U_ref) else 0.0
        log.append((k * dt, s, th1, th2, sdot, th1dot, th2dot, u))

    fig, anim = animate(log, params, s_max=0.18, track_max=0.25, dt=dt, save_path=args.save)

    if not args.no_show:
        plt.show()
