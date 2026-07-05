"""
Animate a closed-loop MPC run (sim_loop.py) -- cart + double pendulum
on the rail, viewed from the side. theta measured from vertical-up, so
link tip = pivot + l*(sin(theta), cos(theta)).

Run directly for a live matplotlib window. Pass --save path.gif to also
write an animated gif (works headless, no display needed).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from sim_loop import run

CART_W, CART_H = 0.12, 0.06


def animate(log, params, s_max=0.23, track_max=0.4, dt=0.05, save_path=None, fps=None):
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
        info_text.set_text(
            f"t={i * dt:5.2f}s  s={S[i]:+.3f}m  th1={TH1[i]:+.2f}  "
            f"th2={TH2[i]:+.2f}  u={U[i]:+.1f}"
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
        m1=0.2, m2=0.15, l1=0.3, l2=0.25, M=1.0,
        I1=0.2 * 0.3 ** 2 / 12, I2=0.15 * 0.25 ** 2 / 12,
    )
    x0 = [0.0, np.pi, np.pi, 0.0, 0.0, 0.0]  # both links hanging down
    dt = 0.05

    log = run(params, x0, Np=20, dt=dt, steps=120, s_max=0.23, u_max=12.0)
    fig, anim = animate(log, params, s_max=0.23, dt=dt, save_path=args.save)

    if not args.no_show:
        plt.show()
