"""
Central configuration: every hyperparameter, range, weight, and path lives
here so the rest of the package has no magic numbers. Pure data + a couple
of small helpers; no heavy logic.

Running convention: the physics modules (dynamics, mpc, sim_loop, ...) live
at the repo root. This module inserts the repo root on sys.path so
`import mpc` works whether you run `python -m pinn.train` or `python
pinn/train.py`.
"""
import os
import sys

# --- make repo-root physics modules importable from inside the package ----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from mpc import MOTOR_FORCE_MAX, MOTOR_FREE_SPEED, Q_DIAG  # noqa: E402  single source of truth

# ---------------------------------------------------------------- paths ---
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
CKPT_DIR = os.path.join(PACKAGE_DIR, "checkpoints")
# v2: cold-solved labels from the Np=40 / Qf=P teacher. seed_dataset.npz (v1)
# is kept for reference only -- its labels depend on solve order, not state.
SEED_DATASET = os.path.join(DATA_DIR, "seed_v2.npz")
NORM_STATS = os.path.join(DATA_DIR, "norm_stats_v2.npz")
RESULTS_DIR = os.path.join(PACKAGE_DIR, "results")

# ------------------------------------------------------------- actuator ---
# Network outputs VOLTAGE in [-V_MAX, V_MAX]; actuator.py maps it to force.
# Supersedes CLAUDE.md's +-12 V note (user-specified 24 V bus, motor
# 57HS82-4008A08-D21).
V_MAX = 24.0

# ------------------------------------------------- fixed system params ----
# Not conditioning inputs: held at nominal for every sampled config.
CART_MASS = 1.0        # M [kg]
M_ENC1 = 0.18          # theta1 encoder, on the cart [kg]
M_ENC2 = 0.18          # theta2 encoder, at the joint [kg]

# ------------------------------------------- conditioning param ranges ----
# ~+-25-30% box around the nominal rig (rod-only masses). full_params_from_ml
# fills the dependents I_i = m_i*l_i**2/12, lc_i = l_i/2.
NOMINAL = dict(m1=0.12, m2=0.09, l1=0.30, l2=0.25)
PARAM_LOW = np.array([0.08, 0.06, 0.22, 0.18])   # [m1, m2, l1, l2]
PARAM_HIGH = np.array([0.16, 0.12, 0.38, 0.32])
PARAM_NAMES = ("m1", "m2", "l1", "l2")

# Wider box for L_EL collocation points (regions the MPC never solves).
COLLOC_PARAM_LOW = np.array([0.06, 0.05, 0.18, 0.15])
COLLOC_PARAM_HIGH = np.array([0.20, 0.14, 0.42, 0.36])

# ---------------------------------------- initial-state sampling box ------
# SMALL perturbations from upright (zeros) -- the MPC's reliable regime.
STATE_PERT = np.array([0.05, 0.15, 0.15, 0.05, 0.30, 0.30])
# Wider state box for L_EL collocation.
COLLOC_STATE_PERT = np.array([0.08, 0.30, 0.30, 0.10, 0.60, 0.60])

# --------------------------------------------------- dataset generation ---
N_CONFIGS = 1024           # distinct pendulum configs (power of 2 for Sobol)
N_STATES_PER_CONFIG = 16   # initial states solved per config. v1 used 200 x 80:
                           # the net memorized each config (train MSE -> 0.08,
                           # val rising from epoch 1) and a linear fit beat it on
                           # held-out configs. Generalizing over (m, l) needs many
                           # configs more than many states per config.
DT = 0.05
MPC_NP = 40                # 2.0 s horizon. 20 (1.0 s) is shorter than the cart-
                           # recentring timescale: 1.6% closed-loop success vs 56%
                           # at 40 (with Qf = LQR P, see dataset.make_teacher).
                           # 50 buys only 56 -> 61% for superlinear solve cost.
S_MAX = 0.18
MAX_LABEL_FACTOR = 1.5     # reject |u| > this * MOTOR_FORCE_MAX
MAX_FAIL_FRAC = 0.20       # drop a config if it fails more than this fraction
SEED = 0

# Evaluation MUST draw configs from a different point in the Sobol stream than
# dataset generation, or the "held-out" interpolation split is not held out.
# Both used default_rng(SEED) -> pu.sample_configs, which made the eval configs
# byte-identical to training configs 0..n-1 and every interpolation number a
# training-set number. Offsetting breaks the stream; the assert in
# evaluate.sample_interp_configs proves it stayed broken-free.
EVAL_SEED_OFFSET = 10_000

# Seed-dataset sampling is a *mixture* of STATE_PERT plus two wider regimes,
# layered on top without touching MPCController: off-center stabilization
# (cart starts away from s=0) and disturbance rejection (sudden velocity
# kick). Both stay well inside COLLOC_STATE_PERT, which the MPC is known
# not to solve reliably -- these are meant to still converge under IPOPT.
S_OFFCENTER_MAX = 0.9 * S_MAX      # margin below the rail so the solver
                                   # isn't starting flush against a bound
PUSH_VEL_PERT = np.array([0.30, 0.70, 0.70])  # sdot, th1dot, th2dot kick --
                                               # sdot stays below
                                               # MOTOR_FREE_SPEED (~0.4),
                                               # angular rates well short of
                                               # swing-up territory
MIX_WEIGHTS = dict(center=0.5, offcenter=0.25, push=0.25)

# ----------------------------------------------------------- network ------
HIDDEN = (128, 128, 64)    # 3 hidden layers
ACTIVATION = "tanh"        # smooth for the physics-rollout backprop

# --------------------------------------------------------- training -------
DEVICE = "cpu"             # float64 3x3 solves + small batches -> CPU
BATCH_SIZE = 256
EPOCHS = 300
LR = 1e-3
LR_MIN = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
VAL_CONFIG_FRAC = 0.15     # fraction of *configs* held out for validation
EARLY_STOP_PATIENCE = 40   # epochs w/o val improvement
EARLY_STOP_MIN_EPOCH_FRAC = 0.70  # matches RAMP_FRAC: L_data plateaus fast
                                  # during warmup (it's the only active term)
                                  # -- without this floor, patience expires
                                  # before L_physics/L_barrier/L_EL ever get
                                  # nonzero weight, silently degrading the
                                  # run to a plain imitation NN

# ------------------------------------------------------ physics rollout ---
PHYS_N = 10                # rollout horizon for L_physics (5-20)
# MPC's own Q diagonal -- weight state deviation the way the teacher does.
# Imported, not copied: losses.L_physics and baselines' LQR both read this, and
# a hand-copied literal here meant editing mpc.py's Q left all three disagreeing.
STATE_COST_W = Q_DIAG
DU_MAX = MOTOR_FORCE_MAX * 0.5   # per-step force-rate barrier threshold [N]
N_COLLOC = 512             # collocation points per L_EL evaluation
EL_EPS = 1e-3              # small V**2 regularizer weight inside L_EL

# --------------------------------------------- loss weights + annealing ---
# Fully-ramped target weights; loss_weights(epoch) in losses.py interpolates
# from data-only warmup -> physics/barrier-heavy.
W_DATA = 1.0
W_PHYS = 1.0
W_BAR = 0.5
W_EL = 0.1
WARMUP_FRAC = 0.20         # data-only fraction of training
RAMP_FRAC = 0.70           # physics/barrier fully ramped in by this fraction

# --------------------------------------------------------------- DAgger ---
DAGGER_ROUNDS = 3
# These are what actually produced the committed dataset_round1.npz (which has
# exactly 24 DAgger config_ids, 200-223), ~25 min/round. The previous defaults
# of 40/10/120/5 were ~98 min/round and were never run -- the Jul 28 round
# monkeypatched these values, so config.py misdescribed the committed artifact.
DAGGER_CONFIGS = 48        # configs rolled out per round (mix train + unseen)
DAGGER_ICS = 6             # initial conditions per config
DAGGER_STEPS = 100         # closed-loop steps per rollout
DAGGER_SUBSAMPLE = 4       # keep every Nth visited state (avoid flooding)


def state_bounds():
    """(low, high) arrays for the initial-state sampling box."""
    return -STATE_PERT, STATE_PERT
