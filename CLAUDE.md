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

### Step 2 — done (teacher rebuilt 2026-09-27)
- `mpc.py`: motor spec is NEMA23 (57HS82) on a **60-tooth** GT2 pulley — `MOTOR_FORCE_MAX` **57.6 N**, `MOTOR_FREE_SPEED` **1.20 m/s**. This supersedes the original 20-tooth spec (173 N / 0.4 m/s). **Speed, not force, was the binding constraint on this plant:** sweeping only the speed ceiling gave 0.4 m/s → 0/3 ICs balanced, 0.8 → 1/3, 1.2 → 3/3. IPOPT tuned with `mu_strategy='adaptive'` + `nlp_scaling_method='gradient-based'`. Dynamics are built as SX `ca.Function`s so the NLP is `expand`ed — cold solves ~3 s instead of ~10 s.
- **The teacher is `dataset.make_teacher` / `dataset.label`**, used by both the seed set and DAgger: `Np = 40` (2.0 s), terminal cost `Qf = P` (LQR cost-to-go), and every label a **cold solve** (`solve(x, cold=True)`, zero initial guess).
- `pinn/param_utils.py`: **scrambled Sobol** sampling of $(m_1,m_2,l_1,l_2)$. Sobol warns unless `n` is a power of two.
- `pinn/dataset.py`: `N_CONFIGS = 1024` × `N_STATES_PER_CONFIG = 16`, parallel across configs; ICs from a centre / off-centre / push **mixture**; `config_id` kept so splits hold out whole configs.
- `pinn/actuator.py`: voltage↔force map (torque-speed derate).

### Step 3 — done
`pinn/model.py`: 10→128→128→64→1, tanh, output `tanh × V_MAX` (voltage, hard-bounded by construction). Normalization stats stored as model buffers (self-contained checkpoint).

### Step 4 — done, all four terms
`pinn/losses.py`: $L_{data}$ (voltage MSE); $L_{physics}$ = **cost-to-go ratio** — the teacher's objective over a 10-step differentiable rollout, $\sum_k x_k^\top Q x_k + R F_k^2$, plus terminal $x_N^\top P x_N$, divided by $x_0^\top P x_0$ (P cached per config); $L_{barrier}$ (soft position/velocity/rate penalties); $L_{EL}$ (Lyapunov-style collocation residual — *not* a literal Euler-Lagrange residual). Annealed weighting: data-only warmup (0–20%) → ramp (20–70%) → physics/barrier-heavy (70–100%).

### Step 5 — done
`pinn/train.py`: Adam + cosine decay, early stopping, val split = **fixed hash of `config_id`** (stable across DAgger rounds), norm stats computed from the dataset being trained. `pinn/dagger.py`: closed-loop rollout (stopped at evaluate's divergence test) → parallel cold-solve relabel → warm-started retrain; `--out-dir/--ckpt-dir` keep runs apart, `--ablation data_only` gives the imitation+DAgger baseline.

### Step 6 — done (2026-09-27)
32 held-out configs × 8 centre ICs × 150 steps = 256 rollouts per cell; standard error ≈ 0.03; retrains of the same recipe vary ≈ ±0.04.

| policy | interp success | interp diverged | extrap success |
|---|---|---|---|
| MPC teacher (cold, Np=40, Qf=P) | **0.785** | 0.211 | — |
| **PINN + DAgger r3** | **0.539** | 0.242 | **0.289** |
| imitation + DAgger r3 | 0.203 | 0.320 | 0.090 |
| LQR | 0.035 | 0.961 | 0.051 |

Every round (`pinn/results/final_v2_20260927_064846.json`): PINN r0–r3 = 0.336 / 0.535 / 0.496 / 0.539; imitation r0–r3 = 0.016 / 0.070 / 0.270 / 0.203. Round 3 is the headline because it was the planned last round — do not cherry-pick the best round on the eval set. Teacher: `pinn/results/teacher_v2_np40_cold_32x8.json`.

Ablation without DAgger (`eval_20260927_043224.json`, interpolation success): data-only 0.07, +physics 0.33, +physics+barrier 0.41, full 0.40. **$L_{physics}$ carries the gain; $L_{EL}$ adds nothing measurable.** Remaining gap to the teacher is *settling* (recentring the cart), not divergence — the PINN diverges about as often as the teacher.

### Step 7 — not started
Rig is built; system-ID, encoder filtering, weight export, and the safety watchdog are not.

## Artifact status (2026-09-27)

| artifact | state |
|---|---|
| `pinn/data/seed_v2.npz` | **current seed set.** 15,836 samples, 1021/1024 configs, 6382 s |
| `pinn/data/v2s_dagger_{full,data_only}/dataset_round3.npz` | cumulative; rounds 1–2 are its `config_id < 1069` / `< 1117` prefixes |
| `pinn/checkpoints/v2/stable/dagger_full/round3_best.pt` | **headline PINN — use as `--full-ckpt`** |
| `pinn/checkpoints/v2/stable/` | seed nets + both DAgger runs, hashed split |
| `pinn/checkpoints/v2/ratio/` | ratio-loss ablation variants (`--ckpt-dir` for `--ablation`) |
| `pinn/checkpoints/v2/*.pt` | old greedy $L_{physics}$; kept only for `eval_20260927_034619.json` |
| `pinn/data/seed_dataset.npz`, `spec60_full.pt`, `round{0,1}_best.pt` | **v1, do not use** — labels depend on solve order (below) |

## What was wrong with v1, and why (2026-09-27)

Every v1 run plateaued at val MSE ≈ 20 V² on held-out configs, and a plain linear regression beat the net (20.75 vs 22.77) while the net fit training configs to 0.08. Causes, in order of impact:

1. **Labels were not a function of the state.** Each IC was warm-started from the previous, *unrelated* IC's solution; the NLP is non-convex, so the same state re-solved warm vs cold differed by 1.7–7.6 N RMS at Np=20 and **9.8–20.1 N at Np=40** (label std ≈ 20 N). Cold solves are bit-reproducible and were the lowest-cost optimum for 78–90% of states (an LQR-rollout seed: 0–10%, since LQR itself diverges here). With v2 labels the same data-only fit reaches val 1.14 vs linear 4.90 (R² 0.98).
2. **Cold solving also makes a better closed-loop teacher**: 0.562 (warm) → 0.750 (cold) on the 8×8 benchmark at Np=40, Qf=P — better than Np=50 warm.
3. **The old $L_{physics}$ was greedy and ~400× the data term**, so physics-trained nets kept the links up and drove the cart off the rail in ~27 steps (0.00 success). The ratio form fixed both.
4. **DAgger's val split reshuffled every round**, leaking trained configs into validation; rounds checkpointed at epoch 1–27, i.e. barely retrained.
5. `weight_overrides` used `w_phys` against config constants named `W_PHYS`, so **every ablation crashed on start** — why no ablation table ever existed.

### Teacher sweep (2026-08-10, warm-started teacher, 8 configs × 8 centre ICs)

| `Np` | `Qf` | `thdot_max` | success | diverged |
|---|---|---|---|---|
| 20 | 10·Q *(generated v1)* | none | 0.016 | 0.906 |
| 20 | P | none | 0.312 | 0.688 |
| 30 | P | none | 0.391 | 0.609 |
| 40 | 10·Q | none | 0.422 | 0.516 |
| 40 | P | 1.0 | 0.312 | 0.688 |
| 40 | P | 1.5 | 0.484 | 0.516 |
| 40 | P | none | 0.562 | 0.438 |
| 50 | P | none | 0.609 | 0.391 |
| 40 | P, **cold** | none | **0.750** | 0.250 |

- **Do not pass `thdot_max`** — it hurts monotonically.
- **`Qf = P` matters independently of horizon** (0.422 → 0.562 at Np=40); `Qf = 10·Q` makes a short-horizon teacher park the cart at `s_max`.
- **Do not pursue cost re-weighting**: the binding limit is the hard box `|s| ≤ s_max`, which no objective term relaxes.
- **Metric caveat:** `success` tests `peak_s <= S_MAX + S_MAX_SLACK` (1e-6); without the slack, rollouts that rode the hard bound failed on a 10-nm IPOPT overshoot (~20 points of success).

**Hazards not visible from the code**: `pinn/train.py` defaults `out_ckpt` to `round0_best.pt`, so a bare `python -m pinn.train` overwrites a tracked artifact — always pass `out_ckpt`. `dagger.py` without `--out-dir/--ckpt-dir` writes the tracked `round{k}_best.pt`. Dataset generation accumulates everything in memory and writes once at the end — pause it (`SIGSTOP`), don't kill it. Cold solves take seconds on off-centre/push ICs versus ~0.3 s near a trajectory; budget ~1 h 45 min for the 1024 × 16 seed set on 7 workers.
