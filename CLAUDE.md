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
- `mpc.py`: motor spec is NEMA23 (57HS82) on a **60-tooth** GT2 pulley — `MOTOR_FORCE_MAX` **57.6 N**, `MOTOR_FREE_SPEED` **1.20 m/s**. This supersedes the original 20-tooth spec (173 N / 0.4 m/s). **Speed, not force, was the binding constraint on this plant:** sweeping only the speed ceiling gave 0.4 m/s → 0/3 ICs balanced, 0.8 → 1/3, 1.2 → 3/3. 57.6 N still accelerates the ~1.5 kg cart at ~38 m/s², far beyond anything it commands. IPOPT tuned with `mu_strategy='adaptive'` + `nlp_scaling_method='gradient-based'` to converge on the badly-scaled NLP (~0.3s warm-started, ~0.027s measured).
- `pinn/param_utils.py`: **scrambled Sobol** sampling of $(m_1,m_2,l_1,l_2)$ (`qmc.Sobol(d=4, scramble=True)`, stratified-uniform fallback), derives $I_i, lc_i$. Not Latin-Hypercube, despite earlier notes here. Sobol warns unless `n` is a power of two — prefer `--n-configs 16`.
- `pinn/dataset.py`: one warm-started `MPCController` per config (avoids ~19s IPOPT cold-start per sample), parallelized across configs via `multiprocessing`; initial states drawn from a **mixture** of three regimes — center (small perturbation), off-center (cart position widened toward the rail), and push (velocity kick) — so the teacher dataset covers off-center stabilization and disturbance rejection, not just regulation from dead-center; `config_id` retained per sample so train/val splits can hold out entire configs
- `pinn/actuator.py`: voltage↔force map (torque-speed derate), since the network outputs voltage but the plant/MPC labels are in force

### Step 3 — done
`pinn/model.py`: 10→128→128→64→1, tanh, output `tanh × V_MAX` (voltage, hard-bounded by construction). Normalization stats stored as model buffers (self-contained checkpoint).

### Step 4 — done, all four terms
`pinn/losses.py`: $L_{data}$ (voltage MSE), $L_{physics}$ (5–10 step differentiable rollout, per-batch-element under its own pendulum config), $L_{barrier}$ (soft position/velocity/rate penalties), $L_{EL}$ (Lyapunov-style collocation residual on random state/param points never solved by MPC — included as core). Annealed weighting: data-only warmup (0–20%) → ramp (20–70%) → physics/barrier-heavy (70–100%).

### Step 5 — done
`pinn/train.py`: Adam + cosine decay, grouped validation split (entire configs held out, not random rows), early stopping. `pinn/dagger.py`: closed-loop PINN rollout → MPC relabels visited states → retrain warm-started, 2–3 rounds; reuses the same off-center/push mixture sampler for initial conditions.

Verified via `pinn/smoke_test.py`: per-module checks pass, and a tiny end-to-end run (98 samples, 5 configs → train → 1 DAgger round) shows val MSE dropping 1.407→0.646 and DAgger adding relabeled points, confirming the pipeline assembles and runs correctly.

### Step 6 — implemented, results not yet trustworthy
`pinn/evaluate.py` has metrics (settling time, peak deviation, control effort, success rate, plus graded survival/peak metrics), `run_ablation`, `run_generalization`, and an MPC-teacher policy so the *teacher* can be scored on the same metric as its students. `pinn/baselines.py` has the LQR gain, the LQR cost-to-go, and the plain-imitation baseline (which is literally the data-only ablation). Results are written to `pinn/results/` on every run.

Not yet done: the three `ablation_*.pt` variants have never been trained (~16 min each), so no ablation table exists.

### Step 7 — not started
Rig is built; system-ID, encoder filtering, weight export, and the safety watchdog are not.

## Artifact status (2026-08-10)

| artifact | state |
|---|---|
| `pinn/data/seed_dataset.npz` | 15,996 samples, 200/200 configs kept, 0 dropped, 4152 s, at the 60-tooth spec |
| `pinn/data/dataset_round1.npz` | seed + 732 DAgger points from 24 configs (`config_id` 200–223) |
| `pinn/checkpoints/spec60_full.pt` | epoch 224, `val_combined` 34.20 — **use this as `--full-ckpt`** |
| `pinn/checkpoints/round1_best.pt` | epoch 182, `val_combined` 23.66 (30.8% better) |
| `pinn/checkpoints/round0_best.pt` | **do not use.** Epoch 26 of 300, inside the data-only warmup, so it *is* a data-only net; also carries pre-60-tooth norm buffers and a 173 N actuator |
| DAgger rounds 2–3 | never ran — the round-1 driver crashed on a stale-docstring unpack after round 1 saved |

**Known-good numbers do not exist yet.** `val_data_mse` never improved in either run (22.77 → 21.94; 18.54 → 21.84 within round 1) — the `val_combined` gain came from the physics/barrier terms, not from better imitation.

### Teacher fix: terminal cost + horizon (measured 2026-08-10)

Success below is the corrected metric — see the note on solver slack after the table.

| `Np` | `Qf` | success | diverged | steps survived | peak θ₁ |
|---|---|---|---|---|---|
| 20 | 10·Q *(what generated the dataset)* | **0.016** | 0.906 | 36/151 | 1.351 |
| 20 | P | 0.312 | 0.688 | 55/151 | 0.917 |
| 30 | P | 0.391 | 0.609 | 68/151 | 0.962 |
| 40 | 10·Q | 0.422 | 0.516 | 81/151 | 0.742 |
| 40 | P | 0.562 | 0.438 | 92/151 | 0.744 |
| 50 | P | **0.609** | 0.391 | 98/151 | 0.527 |

8 configs × 8 centre ICs × 150 steps, held-out configs (`EVAL_SEED_OFFSET`), `pinn/results/eval_*.json`.

Two conclusions:

1. **`Qf = P` (the LQR cost-to-go, `baselines.lqr_cost_to_go`) earns its place independently of horizon** — at a fixed `Np=40` it lifts success 0.422 → 0.562. `Qf = 10*Q` tells the optimizer nothing about what happens after the horizon, so it parks the cart; `P` approximates the true value function and the finite horizon starts behaving like an infinite one.
2. **Horizon returns flatten after `Np=40`** (0.562 → 0.609 for 50), while dataset-generation cost grows superlinearly. `Np=50` clears 0.6; `Np=40` at 0.562 is a defensible cheaper choice, since 0.6 was a chosen gate and not a physical threshold.

**Metric caveat that mattered:** `success` requires `peak_s <= S_MAX`, but the cart bound is a *hard* MPC constraint the optimizer may ride exactly to `S_MAX`, and IPOPT satisfies it only to ~1e-8. Rollouts that settled cleanly and never diverged were being failed for a 10-nanometre overshoot — about 20 points of success across the sweep. `S_MAX_SLACK = 1e-6` fixes it. This was only visible because per-rollout rows are persisted; the summary showed a plain failure with no hint that settling had occurred.

### Original blocker: the MPC teacher itself fails

Measured 2026-08-10: the teacher scores **0% success and 88–94% divergence** under `evaluate.py`'s own metric at `MPC_NP = 20`. At `DT = 0.05` that is a **1.0 s horizon**, shorter than this plant's cart-recentring timescale, so the teacher's converged optimum stabilizes the angles and *parks the cart at `s = s_max`* — verified, `s = 0.1800` held for 9 consecutive steps. Every one of the 15,996 labels encodes that policy, so the student's 0% is faithful imitation of a broken teacher. `Np=40` raises center-IC success to 37.5%.

**Do not pursue cost re-weighting.** The earlier hypothesis (cart position under-weighted vs angle) is refuted three ways: `Qf[0,0] = 500` already charges the parked cart; the binding limit is the *hard* box `|s| ≤ s_max` over the whole horizon, which no objective term relaxes; and the LQR that beats the student uses the identical weights.

Two further independent defects: roughly half the seed dataset (off-center and push ICs) comes from states the teacher cannot stabilize even at `Np=40`; and the interpolation split was byte-identical to training configs 0–9 until `EVAL_SEED_OFFSET` landed, so every generalization number predating that is a training-set number.

**Hazards not visible from the code**: `pinn/train.py` defaults `out_ckpt` to `round0_best.pt`, so a bare `python -m pinn.train` overwrites a tracked artifact. Dataset generation accumulates everything in memory and writes once at the end — pause it (`SIGSTOP`), never kill it. The 20-tooth-era dataset is recoverable with `git show 9510bbe^:pinn/data/seed_dataset.npz`.