"""
Voltage <-> force actuator model.

DESIGN DECISION (user-specified): the network outputs VOLTAGE in
[-V_MAX, V_MAX] (V_MAX = 24 V), not force. The physics stack
(forward_dynamics, MPCController) works in force [N], so this module maps
between the two.

The map reuses the torque-speed derate already in mpc.py: a stepper's
usable force fades to zero as cart speed approaches the free-speed ceiling.

    force = (V / V_MAX) * MOTOR_FORCE_MAX * (1 - (sdot/MOTOR_FREE_SPEED)^2)

So V = +-V_MAX commands the full *available* force at the current speed.
This is more physically honest than a force output: the network can never
command a force the motor can't deliver at that speed.

PROVISIONAL: this is a normalized-command approximation for a stepper,
built from catalog specs. The real voltage->force relationship (winding
Kt/Kb/R, microstepping, closed-loop lag) is identified on the actual rig
at hardware Step 7; swap this module's body then. All sim results are in
consistent units and remain valid.

Both functions are torch-friendly (used inside the differentiable physics
rollout) and also accept plain floats/np arrays.
"""
import torch

from pinn import config as C
from mpc import MOTOR_FORCE_MAX, MOTOR_FREE_SPEED

V_MAX = C.V_MAX


def _derate(sdot, floor):
    """1 - (sdot/free_speed)^2, clamped at `floor` (>=0)."""
    if isinstance(sdot, torch.Tensor):
        d = 1.0 - (sdot / MOTOR_FREE_SPEED) ** 2
        return torch.clamp(d, min=floor)
    d = 1.0 - (sdot / MOTOR_FREE_SPEED) ** 2
    return max(d, floor) if not hasattr(d, "__len__") else d.clip(min=floor)


def voltage_to_force(V, sdot):
    """Map commanded voltage [-V_MAX,V_MAX] + current cart speed -> force [N]."""
    return (V / V_MAX) * MOTOR_FORCE_MAX * _derate(sdot, floor=0.0)


def force_to_voltage(F, sdot):
    """
    Inverse map, for converting MPC force labels to voltage targets. Derate
    floored at a small positive value so the division stays finite near the
    speed ceiling; result clamped to the +-V_MAX bus.
    """
    v = V_MAX * F / (MOTOR_FORCE_MAX * _derate(sdot, floor=1e-3))
    if isinstance(v, torch.Tensor):
        return torch.clamp(v, -V_MAX, V_MAX)
    return float(min(max(v, -V_MAX), V_MAX))
