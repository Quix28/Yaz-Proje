"""
PINN controller package for the cart + double inverted pendulum.

Learns to imitate the MPC *balancing* controller (regulation from small
perturbations near upright -- swing-up is handled separately by
trajectory.py), conditioned on the physical parameters (m1,m2,l1,l2) so a
single trained model generalizes to unseen pendulum configurations.

Modules:
    config       -- all hyperparameters, param ranges, loss weights, paths
    param_utils  -- sample configs, build full/batched param dicts
    actuator     -- voltage <-> force map (network outputs voltage)
    dataset      -- MPC-teacher dataset generation + IO + normalization
    model        -- PINNPolicy network
    losses       -- data / physics / barrier / EL losses + annealing
    train        -- seed-dataset training loop
    dagger       -- DAgger closed-loop relabel + retrain
    smoke_test   -- per-module isolation checks + tiny end-to-end run
"""
