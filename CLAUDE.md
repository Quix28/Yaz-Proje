# Physics-Informed Neural Network (PINN) for Double Inverted Pendulum

## Aim
To develop a physics-informed neural network (PINN) that learns to approximate a Model Predictive Controller for balancing a cart-mounted double inverted pendulum, conditioned on the system's physical parameters (mass and length), such that a single trained model generalizes to previously unseen pendulum configurations without retraining — and to validate this on a real hardware rig[cite: 1].

## Process Pipeline

### Step 1: Model the system
* Derive Euler-Lagrange equations for the cart + double pendulum, symbolically (sympy/casadi), parameterized by $m_1, m_2, l_1, l_2$[cite: 1].
* Implement forward dynamics as a differentiable function (PyTorch/JAX) — same framework you'll train the network in[cite: 1].
* Define state $\mathbf{x}=[s,\theta_1,\theta_2,\dot s,\dot\theta_1,\dot\theta_2]$, control $u \in [-12, 12]V$[cite: 1].
* Set constraints: track $\pm0.4m$, voltage $\pm12V$, plus velocity and control-rate limits[cite: 1].

### Step 2: Build the MPC teacher
* Pick a solver: CasADi+IPOPT or acados (nonlinear MPC), or iLQR if NMPC is too slow/unstable to sample at scale[cite: 1].
* Sample a distribution of $(m_1,m_2,l_1,l_2)$ and initial states (small perturbations from upright)[cite: 1].
* Solve MPC for each sample, log $(\mathbf{x}, m_1,m_2,l_1,l_2) \rightarrow u^*$[cite: 1].
* This is your seed dataset[cite: 1].

### Step 3: Build the network
* Inputs: 6 states + 4 params = 10, normalized/standardized[cite: 1].
* 3–4 dense layers, 64–128 units, tanh/swish[cite: 1].
* Output: single $u$, tanh scaled by 12V[cite: 1].

### Step 4: Loss function
* $L_{data}$: MSE(u_PINN, u_MPC) on the labeled dataset[cite: 1].
* $L_{physics}$: roll out N steps (5–20) through your differentiable dynamics using $u_{PINN}$; penalize deviation from upright over the whole rollout, not just one step[cite: 1].
* $L_{barrier}$: penalize any predicted state in the rollout violating position/velocity/control-rate limits[cite: 1].
* Optionally: sample random collocation points (state/param combos never solved by MPC) and penalize the Euler-Lagrange residual directly there — this is what makes it a real PINN loss, not just imitation + regularization[cite: 1].
* Combine with weights; anneal from data-heavy to physics/barrier-heavy over training[cite: 1].

### Step 5: Train
* Train on seed dataset with combined loss[cite: 1].
* Roll out the trained policy in closed-loop simulation[cite: 1].
* Query the MPC teacher on the new states it visits[cite: 1].
* Add those to the dataset, retrain[cite: 1].
* Repeat 2–3 rounds (this fixes compounding-error drift from pure imitation)[cite: 1].

### Step 6: Evaluate
* Metrics: settling time, peak deviation, control effort, success rate — averaged over many random ICs[cite: 1].
* Ablation: data-only vs +physics vs +physics+barrier — proves the physics loss matters[cite: 1].
* Generalization test: train on a range of $(m,l)$, test on held-out interpolated and extrapolated combos, compare against a plain imitation NN and an LQR baseline — this is your novelty claim[cite: 1].
* If any of this fails to show the physics-informed version winning, that's the finding — report it honestly[cite: 1].

### Step 7: Hardware deployment
* System-identify your actual rig: real $m,l$, friction/damping — don't assume frictionless[cite: 1].
* Filter velocity estimates from encoders (raw finite-difference is noisy)[cite: 1].
* Export weights (TFLite Micro or raw C++ header); test post-quantization behavior if quantizing[cite: 1].
* Add an independent hardware watchdog that cuts power on constraint violation, separate from the network[cite: 1].
* Run closed-loop on the real rig, compare against sim results[cite: 1].

## Implementation Status

Locked decisions made with the user before implementation: network output is **voltage** (tanh × V_MAX, V_MAX = 24V — supersedes the ±12V note above, since the user specified a 24V bus with a 57HS82-4008A08-D21 NEMA23 stepper motor), core training scope first (Steps 2–5), evaluation/ablation/LQR baseline (Step 6) deferred to a second pass, and $L_{EL}$ included as core rather than optional. All new code lives in the `pinn/` package; existing physics files (`dynamics.py`, `sim_loop.py`, `trajectory.py`) are untouched except the Step 0 motor-spec fix in `mpc.py`.

### Step 1 — done (pre-existing)
Differentiable dynamics already implemented in `dynamics.py` before this work began.

### Step 2 — done
- `mpc.py`: motor spec restored to NEMA23 (57HS82, ~165N) after a bad-merge regression to NEMA17; IPOPT tuned with `mu_strategy='adaptive'` + `nlp_scaling_method='gradient-based'` to converge on the resulting badly-scaled, over-actuated NLP (30/30 solves, ~0.3s warm-started)
- `pinn/param_utils.py`: Latin-Hypercube sampling of $(m_1,m_2,l_1,l_2)$, derives $I_i, lc_i$
- `pinn/dataset.py`: one warm-started `MPCController` per config (avoids ~19s IPOPT cold-start per sample), parallelized across configs via `multiprocessing`; initial states drawn from a **mixture** of three regimes — center (small perturbation), off-center (cart position widened toward the rail), and push (velocity kick) — so the teacher dataset covers off-center stabilization and disturbance rejection, not just regulation from dead-center; `config_id` retained per sample so train/val splits can hold out entire configs
- `pinn/actuator.py`: voltage↔force map (torque-speed derate), since the network outputs voltage but the plant/MPC labels are in force

### Step 3 — done
`pinn/model.py`: 10→128→128→64→1, tanh, output `tanh × V_MAX` (voltage, hard-bounded by construction). Normalization stats stored as model buffers (self-contained checkpoint).

### Step 4 — done, all four terms
`pinn/losses.py`: $L_{data}$ (voltage MSE), $L_{physics}$ (5–10 step differentiable rollout, per-batch-element under its own pendulum config), $L_{barrier}$ (soft position/velocity/rate penalties), $L_{EL}$ (Lyapunov-style collocation residual on random state/param points never solved by MPC — included as core). Annealed weighting: data-only warmup (0–20%) → ramp (20–70%) → physics/barrier-heavy (70–100%).

### Step 5 — done
`pinn/train.py`: Adam + cosine decay, grouped validation split (entire configs held out, not random rows), early stopping. `pinn/dagger.py`: closed-loop PINN rollout → MPC relabels visited states → retrain warm-started, 2–3 rounds; reuses the same off-center/push mixture sampler for initial conditions.

Verified via `pinn/smoke_test.py`: per-module checks pass, and a tiny end-to-end run (98 samples, 5 configs → train → 1 DAgger round) shows val MSE dropping 1.407→0.646 and DAgger adding relabeled points, confirming the pipeline assembles and runs correctly.

### Step 6 — deferred (explicit second pass)
`evaluate.py` (settling time, peak deviation, control effort, success rate; ablation; interpolation/extrapolation generalization split) and `baselines.py` (LQR + plain imitation NN) not yet started.

### Step 7 — not started
Pending real hardware access.

**Ready for**: full-scale seed dataset generation (~200 configs × 80 states, 30–60 min), full training (300 epochs, ~1–2hrs CPU), 3 DAgger rounds.