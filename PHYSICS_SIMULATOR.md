# Physics Simulator — Reference Documentation

This document covers the **physics/control stack** of the project: the cart +
double-inverted-pendulum model, its symbolic derivation, the MPC controller,
the closed-loop simulator, the swing-up controllers, and the visualizer. This
is the pre-existing simulation code that the PINN (`pinn/`) is trained
against — it is documented separately here because it stands on its own as a
simulator/controller stack, independent of the neural network work.

Files covered (project root):

| File | Role |
|---|---|
| [`derive_eom.py`](derive_eom.py) | Symbolic (SymPy) derivation of the equations of motion — a cross-check, not used at runtime |
| [`dynamics.py`](dynamics.py) | Closed-form forward dynamics, PyTorch — the "plant" / source of truth |
| [`mpc.py`](mpc.py) | Nonlinear MPC controller (CasADi + IPOPT) — the "teacher" controller |
| [`sim_loop.py`](sim_loop.py) | Closed-loop simulation: MPC senses/solves, `dynamics.py` steps the plant |
| [`swingup.py`](swingup.py) | Online energy-shaping swing-up controller, hands off to MPC near upright |
| [`trajectory.py`](trajectory.py) | Offline swing-up trajectory via direct collocation + TVLQR tracking |
| [`visualize.py`](visualize.py) | Matplotlib animation of a swing-up run |

Everything here operates on **force** as the control input (`u`, in Newtons
on the cart), not motor voltage. The voltage→force actuator model lives in
`pinn/actuator.py` and is a separate concern layered on top for the PINN /
real hardware.

---

## 1. The physical system

A cart rides on a horizontal rail. Two pendulum links are chained from the
cart: link 1 pivots on the cart, link 2 pivots on the tip of link 1
("double pendulum" in series, not two independent pendulums on the same
cart).

**State vector** (6-dimensional, used identically in every file):

```
x = [s, th1, th2, sdot, th1dot, th2dot]
```

- `s` — cart position along the rail [m]
- `th1` — angle of link 1, measured from **vertical-up** [rad]
- `th2` — angle of link 2, measured from **vertical-up** [rad]
- `sdot, th1dot, th2dot` — corresponding velocities

Both angles are zero when both links point straight up (the target/upright
equilibrium). `th1 = th2 = π` is both links hanging straight down (the
starting point for swing-up).

**Control input**: `u`, a scalar net force on the cart [N]. Positive/negative
sign convention follows the derivation in `derive_eom.py`/`dynamics.py`
directly (force pushes the cart in the `+s` direction).

**Parameters dict** — every file takes physical parameters as a plain Python
`dict` with the same key set (this is the shared "language" between all the
physics files):

| Key | Meaning | Default (where optional) |
|---|---|---|
| `m1`, `m2` | link (rod-only) mass [kg] | required |
| `l1`, `l2` | link length, pivot-to-pivot [m] | required |
| `I1`, `I2` | link moment of inertia about its own COM [kg·m²] | required (typically `m*l²/12`, uniform rod) |
| `M` | cart mass [kg], **not** including the theta1 encoder | required |
| `lc1`, `lc2` | pivot-to-COM distance [m] | `l_i / 2` |
| `g` | gravity [m/s²] | `9.81` |
| `m_enc1` | mass of the theta1 encoder, mounted on the cart | `0.18` |
| `m_enc2` | mass of the theta2 encoder, mounted at the link1/link2 joint | `0.18` |
| `b0, b1, b2` | viscous damping — cart, joint1, joint2 | `0.02, 0.0008, 0.0008` |
| `cf0, cf1, cf2` | Coulomb (dry) friction magnitude — cart, joint1, joint2 | `0.05, 0.0025, 0.0025` |

Two encoder-mass details matter for reading the mass-matrix code correctly:
- `m_enc1` rides on the cart and only translates → folds straight into an
  effective cart mass (`M_eff = M + m_enc1`).
- `m_enc2` is mounted rigidly at the link1/link2 joint, so it rotates with
  `th1` only (no `th2` dependence) → it folds into the "link-1-arm" mass
  terms (`m2_l1 = m2 + m_enc2`) and into link 1's gravity term, but leaves
  link 2's equation and all `lc2` terms untouched.

These are not simplifications of convenience — the comments in `dynamics.py`
note this was verified symbolically via Euler-Lagrange, not just
pattern-matched in by analogy.

---

## 2. `derive_eom.py` — symbolic derivation (cross-check only)

**Purpose**: derive the equations of motion symbolically with SymPy and print
them, to sanity-check the *structure* of the closed-form equations hand-coded
in `dynamics.py`. It is a standalone script (`python derive_eom.py`), not
imported anywhere else in the codebase.

Deliberately a **skeleton**, not the full model: it omits encoder masses and
Coulomb friction (the `tanh(qdot/eps)` term). Including them would clutter
the symbolic derivation without changing what's being checked — whether the
Euler-Lagrange procedure (Lagrangian → equations of motion) produces the same
*shape* of equations as the hardcoded ones. Treat `dynamics.py` as the source
of truth for the full model; this file only validates the skeleton.

Walkthrough of the script body (no functions — it's a flat script):

1. Declares `t` as the time symbol and `s(t)`, `theta1(t)`, `theta2(t)` as
   SymPy `Function`s of time (so they can be differentiated symbolically).
2. Declares physical constants as positive symbols: `m1, m2, M, l1, l2, lc1,
   lc2, I1, I2, g, u, b0, b1, b2`.
3. Builds Cartesian positions of each link's center of mass from the
   generalized coordinates:
   - `x1 = s + lc1*sin(th1)`, `y1 = lc1*cos(th1)` (link 1 COM)
   - `x2 = s + l1*sin(th1) + lc2*sin(th2)`, `y2 = l1*cos(th1) + lc2*cos(th2)`
     (link 2 COM, chained off link 1's pivot)
4. Differentiates those positions w.r.t. time to get COM velocities.
5. Builds kinetic energy `T` (cart translation + each link's COM translation
   + each link's rotation about its own COM) and potential energy `V`
   (`m*g*y` for each link).
6. `L = T - V` — the Lagrangian, simplified.
7. Builds a dict `Q` of generalized (non-conservative) forces: `u - b0*sdot`
   on the cart coordinate, `-b1*th1dot` and `-b2*th2dot` on the two joint
   coordinates (pure viscous damping resisting each generalized velocity).
8. For each generalized coordinate, computes the Euler-Lagrange equation
   `d/dt(∂L/∂q̇) - ∂L/∂q - Q = 0`, simplifies it, and prints it with
   `sp.pprint`.

Running this script produces three printed equations (`s`, `theta1`,
`theta2`) that you can visually compare against the mass-matrix/RHS terms in
`dynamics.py` and `mpc.py` to confirm they encode the same physics.

---

## 3. `dynamics.py` — the plant (forward dynamics)

**Purpose**: the authoritative closed-form forward-dynamics model, in
PyTorch, used as the actual "physics engine" everywhere a plant needs to be
stepped forward (`sim_loop.py`, `swingup.py`, `trajectory.py`'s
`track_trajectory`, and the PINN's rollout losses). This is the file
`derive_eom.py` cross-checks against, and the file `mpc.py` reimplements in
CasADi symbolics (kept manually in sync — see the module docstring's warning
about `FRICTION_EPS`).

### Module-level constant

**`FRICTION_EPS = 0.05`**
Smoothing width for a `tanh(qdot/eps)` approximation to `sign(qdot)` inside
the Coulomb friction term. Real dry friction is discontinuous at zero
velocity, which breaks gradients (needed for the PINN) and solver
convergence (needed for IPOPT in `mpc.py`); `tanh` keeps it smooth while
still closely approximating a hard sign away from `qdot=0`. The value `0.05`
specifically (rather than something sharper) was chosen because a tighter
kink made the swing-up NLP in `mpc.py`/`trajectory.py` too stiff for IPOPT —
the trajectory crosses zero velocity many times while pumping energy, and
each crossing hit the near-vertical edge of a sharper tanh, exhausting the
solver's iteration budget. **`mpc.py` uses the identical constant** so the
solver's internal model of the plant stays bit-consistent with this file.

### `forward_dynamics(state, u, params) -> xdot`

The only function in the file.

- **Inputs**:
  - `state`: `(..., 6)` tensor, the state vector above (batchable — leading
    dims can be anything, e.g. a batch of rollouts, each under its own
    pendulum configuration).
  - `u`: `(...,)` tensor, force on the cart [N].
  - `params`: dict as described in §1 (values must broadcast against
    `state.shape[:-1]`, so this supports per-batch-element physical
    parameters — this is what lets the PINN's physics loss roll out many
    different pendulum configs in one batched call).
- **Returns**: `(..., 6)` tensor, `xdot` — the time-derivative of the state,
  i.e. `[sdot, th1dot, th2dot, sddot, th1ddot, th2ddot]`.

**What it does, step by step**:

1. Unpacks the 6 state components via `torch.unbind`.
2. Reads all physical parameters out of `params`, applying the defaults from
   §1 for anything optional (`g`, damping, Coulomb friction, encoder
   masses, COM offsets).
3. Computes friction torques/forces for each of the 3 generalized
   coordinates as `viscous * velocity + coulomb * tanh(velocity / FRICTION_EPS)`.
4. Computes `sin`/`cos` of `th1`, `th2`, and `th1 - th2` (the last for the
   coupling terms between the two links).
5. Folds encoder masses into effective masses: `M_eff = M + m_enc1` (cart +
   translating encoder), `m2_l1 = m2 + m_enc2` (link-2 mass as seen from the
   link-1-arm terms, since encoder2 rotates with `th1`).
6. Assembles the symmetric 3×3 mass matrix `Mmat` (entries `M11..M33`) via
   `torch.stack`, batched over the leading dimensions.
7. Assembles the right-hand-side vector `rhs` — generalized forces from
   control input, friction, centrifugal/Coriolis terms (`sin(th)*thdot²`),
   and gravity (`g*sin(th)`), one row per generalized coordinate.
8. Solves the linear system `Mmat @ qddot = rhs` with `torch.linalg.solve`
   (batched), giving `[sddot, th1ddot, th2ddot]`.
9. Returns the full derivative `[sdot, th1dot, th2dot, sddot, th1ddot,
   th2ddot]` by concatenating the velocities (already known — they're just
   copied from the input state) with the freshly solved accelerations.

This function has no explicit integrator — callers (`sim_loop.py`,
`swingup.py`, `trajectory.py`) each implement their own RK4 step by calling
`forward_dynamics` four times per step (see §7, the RK4 pattern).

---

## 4. `mpc.py` — the MPC teacher controller

**Purpose**: a nonlinear model-predictive controller for upright
regulation, built with CasADi + IPOPT. This is the "teacher" that the PINN
imitates (per `CLAUDE.md` Step 2), and is also used directly as the
stabilizing controller in `sim_loop.py` and as the catch/stabilize phase in
`swingup.py`.

### Module-level constants — motor/actuator envelope

These derive the **force** limits used throughout the simulator from a
concrete motor spec (NEMA23 stepper `57HS82-4008A08-D21`, GT2-belt
direct-drive, no gearbox):

- `NEMA23_HOLDING_TORQUE = 2.2` N·m — catalog spec.
- `GT2_PULLEY_RADIUS = 0.006366` m — 20-tooth GT2 pulley pitch radius.
- `DYNAMIC_DERATE = 0.5` — steppers lose torque with speed and can skip
  steps if driven continuously at rated holding torque, so only half the
  holding torque is treated as reliably usable.
- `MOTOR_FORCE_MAX = NEMA23_HOLDING_TORQUE * DYNAMIC_DERATE / GT2_PULLEY_RADIUS`
  ≈ 173 N — the hard force ceiling used as `u_max` everywhere by default.
- `MOTOR_MAX_RPM = 600` — a practical ceiling for *reliable, in-torque*
  operation with a common stepper driver (A4988/DRV8825-class), not the
  motor's theoretical no-load top speed.
- `MOTOR_FREE_SPEED = MOTOR_MAX_RPM * 2π/60 * GT2_PULLEY_RADIUS` ≈ 0.4 m/s —
  that RPM translated through the pulley into a linear cart-speed ceiling.
  Available force is derated toward zero as the cart approaches this speed
  (see `u_avail_k` below) rather than clamped as flat until this line.

`FRICTION_EPS = 0.05` — identical to `dynamics.py`, and must stay identical:
any mismatch here would make the MPC's internal model of the plant diverge
from the actual plant it's controlling, corrupting the reference/tracking
consistency that `trajectory.py`'s TVLQR relies on.

### `_dynamics(x, u, p) -> xdot` (private)

CasADi-symbolics reimplementation of `dynamics.py`'s `forward_dynamics`,
term-for-term identical (same mass matrix entries `M11..M33`, same
friction/RHS formulas), but built with `casadi` ops (`ca.cos`, `ca.solve`,
`ca.vertcat`/`horzcat`) instead of `torch` ops, because IPOPT needs a CasADi
symbolic graph to differentiate through. Operates on unbatched CasADi
`MX`/`SX` vectors (a single state/control at a time — batching happens over
NLP *time steps*, not over vectorized inputs like `dynamics.py`).

### `_rk4_step(x, u, p, dt) -> x_next` (private)

Standard explicit 4th-order Runge-Kutta integrator, one step, built from
four calls to `_dynamics`. This is baked directly into the NLP's equality
constraints (multiple shooting — see below), not called externally.

### `class MPCController`

Finite-horizon nonlinear MPC for upright regulation via multiple shooting.
Builds the CasADi NLP **once** in `__init__`; each control step calls
`.solve(x0)`, which is cheap because it's warm-started from the previous
solution.

#### `__init__(self, params, Np=20, dt=0.05, Q=None, R=1e-2, Qf=None, s_max=0.18, u_max=MOTOR_FORCE_MAX, sdot_max=None, thdot_max=None, sddot_max=None)`

- `Np` — prediction horizon length (steps), `dt` — step size [s].
- `s_max=0.18` m — rail travel bound. Comment explains the derivation: a
  0.5 m MGN12 rail (±0.25 m from center) minus ~8 cm cart footprint minus a
  3 cm safety buffer per side.
- `sddot_max` — if not given, derived as `MOTOR_FORCE_MAX / total_mass`
  where `total_mass = M + m1 + m2`: the maximum acceleration the motor
  could impart if it were the only force acting on the whole assembly.
  Enforced as a **separate** constraint from the force bound because
  pendulum coupling terms (centrifugal/gravity reaction transmitted through
  the links) can push actual cart acceleration above what force-over-mass
  alone would suggest.
- `sdot_max` — defaults to `MOTOR_FREE_SPEED`; available force is derated
  to zero as this is approached rather than staying flat at `u_max`
  (steppers lose usable torque with speed).
- `Q` — default `diag([50, 200, 200, 1, 5, 5])`, i.e. angle error penalized
  much more heavily than cart position, velocity terms lightly. `Qf =
  10*Q` by default (heavier terminal penalty). `R=1e-2` — control effort
  penalty.

**Building the NLP** (inside `__init__`):

- Decision variables: `X` (state trajectory, `NX × (Np+1)`), `U` (control
  trajectory, `1 × Np`).
- Parameters (values plugged in at `.solve()` time without rebuilding the
  NLP): `x0_param` (current state), `xref_param` (target state).
- Cost `J`: for each step `k`, quadratic tracking cost `dx.T @ Q @ dx +
  R*u_k²` plus, after the loop, a terminal cost `dxN.T @ Qf @ dxN`.
- **Multiple-shooting dynamics constraint**: `X[:,k+1] == _rk4_step(X[:,k],
  U[0,k], params, dt)` for every step — this is what makes the predicted
  trajectory dynamically consistent.
- **Cart acceleration bound**: `-sddot_max <= xdot_k[3] <= sddot_max` at
  every step, computed from `_dynamics` directly (not from the RK4-stepped
  state).
- **Torque-speed-derated force bound**: `u_avail_k = u_max * max(0, 1 -
  (sdot/sdot_max)²)` — a crude linear stepper torque-speed curve, then
  `U[0,k]` is bounded to `±u_avail_k`. The `fmax(0, ...)` floor exists
  specifically because an *intermediate* IPOPT iterate (not yet converged)
  can transiently have `|sdot| > sdot_max`, which without the floor would
  make `u_avail_k` negative and turn `opti.bounded(-u_avail_k, U,
  u_avail_k)` into an empty (infeasible) interval — undefined for an
  interior-point barrier method.
- Initial condition constraint: `X[:,0] == x0_param`.
- Rail bound: `-s_max <= X[0,:] <= s_max`.
- Velocity bound: `-sdot_max <= X[3,:] <= sdot_max`.
- Optional angular-rate bounds if `thdot_max` is given.

**Solver options**: IPOPT with `mu_strategy='adaptive'` and
`nlp_scaling_method='gradient-based'`. The comment explains why: the NEMA23
force ceiling (~173 N) is large relative to this cart's tight
position/velocity bounds, so the regulation NLP is over-actuated and badly
scaled — the default monotone barrier strategy stalls
(`Maximum_Iterations_Exceeded` even at 5000 iterations). The adaptive
barrier + gradient-based scaling combination converges in a few hundred
iterations instead. `acceptable_tol`/`acceptable_iter` let IPOPT certify a
near-optimal point early rather than grinding for the last digits of KKT
residual.

`self._X_prev`, `self._U_prev` are initialized to zeros — the warm-start
buffers used by `.solve()`.

#### `solve(self, x0, xref=None) -> (u0, X_pred, U_pred)`

- `x0`: current 6-vector state. `xref`: target state, defaults to the
  all-zeros upright/centered state.
- Sets the NLP's `x0_param`/`xref_param` values, seeds `X`/`U` from the
  previous solve's stored trajectories (`set_initial`), and calls
  `opti.solve()`.
- Extracts the solved `X`/`U` trajectories.
- **Warm-start shift**: stores `X_sol` shifted left by one column
  (dropping the now-past first column, duplicating the last column to fill
  the new tail) into `self._X_prev`, same for `U`. This is what makes
  repeated `.solve()` calls in a receding-horizon loop fast — each solve
  starts near the previous plan shifted by one step, which is close to the
  new optimum.
- Returns `u0` = the first control action to actually apply (receding
  horizon principle — only `u0` is used, everything else is replanned next
  step), plus the full predicted state/control trajectories for
  inspection/logging.

### `if __name__ == "__main__":` block

Demo: builds a controller for a concrete parameter set (encoder masses
tracked separately from link masses, per §1), starts from both links
hanging straight down (`th1=th2=π`), and calls `.solve()` in a loop for 120
steps, feeding each step's predicted next state back in as the new current
state (i.e., this demo trusts the MPC's own internal model as the plant —
it does **not** use `dynamics.py`; that composition is what `sim_loop.py`
does instead). Prints a table of `s, th1, th2, u` per step. Note this demo
actually starts from the *hanging-down* configuration, which is far outside
the small-perturbation regime this regulation-only MPC is designed for — it
mostly demonstrates the API, not a working swing-up (that's what
`swingup.py`/`trajectory.py` are for).

---

## 5. `sim_loop.py` — closed-loop simulator (MPC + independent plant)

**Purpose**: wires `mpc.py`'s controller to `dynamics.py`'s plant as two
genuinely separate components — sense → solve → apply → step → repeat —
rather than trusting the MPC's own internal RK4 prediction as ground truth
(as `mpc.py`'s own demo does). The module docstring is explicit about why
this separation matters: today both the MPC's internal model and the plant
use identical equations and parameters, so no mismatch is visible yet, but
this is the seam where you'd later inject real-rig parameter error, sensor
noise, or unmodeled friction without touching the controller itself.

### `_rk4_step_torch(state, u, params, dt) -> state_next` (private)

Same 4th-order RK4 pattern as `mpc.py`'s `_rk4_step`, but built on
`dynamics.py`'s `forward_dynamics` (PyTorch) instead of CasADi. This is the
"plant-side" integrator, kept as its own copy rather than shared with
`mpc.py`'s because it operates on `torch.Tensor` rather than CasADi
symbolics.

### `run(params, x0, Np=20, dt=0.05, steps=120, s_max=0.18, u_max=MOTOR_FORCE_MAX) -> log`

- Builds one `MPCController` for the whole run.
- Loop, `steps` times:
  1. **Sense → Solve**: `ctrl.solve(x)` on the current numpy state `x`,
     yielding `u0` (only the first control action).
  2. **Apply u0 → step plant**: converts state/`u0` to `torch` tensors,
     steps the plant one `dt` via `_rk4_step_torch` (using `dynamics.py`,
     *not* the MPC's own internal model).
  3. Converts the result back to numpy, appends `(step_index, *state,
     u0)` to `log`.
- Returns `log`, a list of tuples: `(i, s, th1, th2, sdot, th1dot, th2dot,
  u0)`.

### `if __name__ == "__main__":` block

Same demo parameter set as `mpc.py`, same hanging-down start, runs `run()`
for 120 steps, prints every 5th row (plus everything past step 100) as a
table.

---

## 6. `swingup.py` — online energy-shaping swing-up + MPC handoff

**Purpose**: a two-phase **online/reactive** controller — an energy-shaping
swing-up law to pump the pendulum from hanging-down up near vertical, then
handoff to `mpc.py`'s `MPCController` for the final catch and stabilization.
The module docstring explains the motivating finding: an "energy budget"
check confirmed the motor has plenty of energy headroom to swing the
pendulum up, but a short-horizon MPC can't "see" a multi-second pumping
strategy far enough ahead and just gives up near the bottom — hence the
classic two-phase fix (energy-shaping to get near vertical, then
MPC/LQR-style capture, which was already validated to work from small
perturbations).

The energy relation exploited: because `u` only does work through the
cart's own coordinate, `dE/dt = u*sdot - (damping losses)` **exactly**,
regardless of how the two pendulum links are coupled to each other. This
gives a simple, general pump rule — push in the direction of cart motion to
add energy, oppose it to remove energy — with no need for a
double-pendulum-specific heuristic. (This is only overridden near the
physical rail limits, where avoiding a crash takes priority.)

### `_mass_matrix(th1, th2, p) -> 3x3 ndarray` (private)

Numpy reimplementation of the same 3×3 mass matrix as `dynamics.py`/
`mpc.py`'s `_dynamics`, including encoder masses. The docstring flags that
it must stay bit-consistent with the plant, or the energy-shaping law below
would compute the wrong target energy `E_target`.

### `_potential(th1, th2, p) -> float` (private)

Total gravitational potential energy as a function of both link angles.
Encoder1 contributes nothing (its height is constant as it translates with
the cart). Encoder2 rotates with `th1` at radius `l1`, so it contributes a
`m_enc2 * g * l1 * cos(th1)` term, structurally identical to how `m2`'s own
attachment to link 1 contributes.

### `energy(state, p) -> float`

Total mechanical energy `KE + PE` for a given state: kinetic energy via the
quadratic form `0.5 * qdot @ Mmat @ qdot` (using `_mass_matrix`), plus
`_potential`.

### `wrapped_angle(theta) -> float`

Wraps an angle into `(-π, π]`, representing "signed distance to the nearest
upright" — used so the swing-up controller's "near upright" check and any
angle-based logic isn't confused by the pendulum having wound through extra
full rotations during pumping.

### Module constant: `SPEED_GOVERNOR_FRAC = 0.85`

Caps the swing-up's commanded cart speed at 85% of the motor's free-speed
ceiling (`MOTOR_FREE_SPEED`), rather than letting it approach 100% (where
available force — and thus any ability to recover — is exactly zero). The
comment notes this value was tuned: an earlier, tighter `0.6` cap killed
pump amplitude (too conservative), while no cap at all let a strong pump
overshoot into the zero-force region near free-speed.

### `swingup_control(state, p, s_max, E_target, k_E=300.0, k_center=5.0) -> u`

The core reactive control law. Returns a force command:

```
u_pump   = -k_E * (E - E_target) * sdot
u_center = -k_center * s
u        = clip(u_pump + u_center, -u_avail, u_avail)
```

- **Continuous energy-shaping pump**: proportional to both the energy
  error and the cart's own velocity. This naturally backs off as `E`
  approaches `E_target` — unlike a fixed-magnitude bang-bang pump, which
  keeps demanding maximum force in the current direction of motion
  regardless of how close to target energy it already is, and can drive
  cart speed straight past the free-speed ceiling with no way to recover
  once force derates to zero there.
- **Gentle centering bias** (`u_center`) discourages the cart from
  drifting toward a rail limit purely as a byproduct of pumping.
- **Two safety layers**, checked *before* the pump law is even evaluated:
  1. **Speed governor**: if `|sdot| > SPEED_GOVERNOR_FRAC * MOTOR_FREE_SPEED`,
     immediately command maximum available braking force opposing the
     current velocity, overriding the pump law entirely.
  2. **Predictive rail brake**: estimates stopping distance
     `sdot² / (2 * a_brake_ref)` using a *fixed* reference deceleration —
     the force available **at the governor speed ceiling**, not the
     instantaneous (possibly near-zero) force right as that ceiling is
     approached. The docstring notes this fixes an earlier bug where using
     the instantaneous available force was self-referential and
     underestimated the true stopping distance. If projected stopping
     position would exceed `s_max`, command full braking force toward the
     center immediately.
- `u_avail` (force actually available at the current speed, via the same
  torque-speed derate formula as `mpc.py`) bounds the final clipped output.

### `near_upright(state, angle_tol=0.45, rate_tol=2.0) -> bool`

True when both wrapped angles are within `angle_tol` radians of upright and
both angular rates are within `rate_tol` — the handoff condition from
swing-up to MPC.

### `_rk4(state, u, params, dt) -> state_next` (private)

Same RK4-via-`forward_dynamics` pattern as `sim_loop.py`'s integrator, with
one addition: after stepping, it hard-clamps cart velocity to `±1.05 *
MOTOR_FREE_SPEED`. The comment frames this as a stand-in for a real
driver's current-limit/stall protection — a safety backstop against any
controller imperfection driving speed past what the motor can physically
do, not a substitute for the governor/brake logic above (which is meant to
prevent ever reaching this clamp).

### `run(params, x0, swingup_dt=0.002, mpc_dt=0.05, swingup_time=20.0, mpc_steps=300, s_max=0.18, Np=20) -> (log, caught)`

Two phases:

**Phase 1 — swing-up**, at a much finer timestep (`swingup_dt=0.002` vs.
MPC's `0.05`). The docstring explains why the timestep must be this much
finer here specifically: at max acceleration (`~F_max/mass`), a single
0.02–0.05 s step could swing cart velocity clean through the entire
free-speed range before this simple reactive law gets a chance to respond.
The MPC phase can afford a slower loop because IPOPT plans several steps
ahead *inside* each solve; this purely reactive law needs a fast loop, the
same way a real embedded controller running this kind of law would.

- Both links start exactly at the stable hanging equilibrium (zero
  velocity, zero net torque) — an exact fixed point of the dynamics, so a
  short priming kick (`PRIME_STEPS` ≈ 0.2 s of `PRIME_FORCE = 0.3 *
  MOTOR_FORCE_MAX`) is applied first to break the symmetry, since the
  energy-shaping law has nothing to act on with `sdot` exactly zero.
- After priming, steps `swingup_control` + `_rk4` in a loop, checking
  `near_upright` each iteration; breaks out and marks `handed_off = True`
  as soon as it's near upright.
- If the loop exhausts `swingup_time` without reaching the catch basin,
  returns `(log, False)` — swing-up failed to converge in the time budget.

**Phase 2 — MPC catch/stabilize**, only reached if Phase 1 succeeded:

- Computes `th1_ref`/`th2_ref` as the nearest multiple of `2π` to the
  swing-up's *unwrapped* final angles (not literal zero) — because MPC's
  quadratic cost has no notion of angular periodicity, so if the swing-up
  wound through one or more extra full rotations while pumping, targeting
  raw `0.0` would be unreachable/wrong; targeting the nearest equivalent
  angle gives the MPC a reachable setpoint.
- Builds a fresh `MPCController` and runs `mpc_steps` receding-horizon
  steps against `xref`, stepping the plant with the same `_rk4` (now at
  the coarser `mpc_dt`).
- Every step, appends `(t, *state, u, mode)` to `log`, where `mode` is
  the string `"swingup"` or `"mpc"` — this is what lets `visualize.py`
  color/label the two phases distinctly if it inspects that field.

Returns `(log, True)` on success.

### `if __name__ == "__main__":` block

Runs the full two-phase controller from hanging-down, prints whether it
caught the pendulum, the max `|s|` excursion over the whole run, then a
sparse (every 200th row) table of the trajectory plus the final 15 rows in
full.

---

## 7. `trajectory.py` — offline swing-up via direct collocation + TVLQR

**Purpose**: a different, more reliable way to generate a swing-up, framed
as an **offline boundary-value problem** rather than a receding-horizon or
reactive controller. The module docstring explains why both alternatives in
this codebase fell short for the swing-up itself:

- `mpc.py`'s receding-horizon MPC has too short a horizon to "see" the
  value of a multi-second pumping maneuver (same limitation noted in
  `swingup.py`).
- `swingup.py`'s energy-shaping heuristic pumps *total* energy toward the
  upright value, but for a double pendulum, matching total energy doesn't
  uniquely determine the configuration — the system can wander the
  constant-energy shell indefinitely without ever landing both links
  upright simultaneously.
- Warm-starting a long-horizon MPC with the heuristic's own trajectory
  doesn't fix this either: IPOPT is a *local* optimizer, so it just drops
  into whatever non-convex "valley" that trajectory already sits in — it
  won't jump to a different valley on its own.

**The fix used here**: hard-constrain both trajectory endpoints exactly
(hanging → upright, both at rest) and minimize control effort only, then
seed IPOPT with a naive straight-line interpolation between the two states
in state space. That initial guess completely ignores dynamics (not a valid
trajectory at all), but it starts IPOPT in the *topologically correct*
valley — the one that actually connects bottom to top — and IPOPT bends it
into a dynamically consistent path from there. The resulting reference
trajectory is then tracked in closed loop with a short-horizon TVLQR
controller (for robustness to model mismatch/disturbance) rather than
replayed open-loop.

### `solve_swingup_trajectory(params, x0, xf, Np=150, dt=0.05, s_max=0.18, max_iter=3000) -> (X_opt, U_opt, success)`

Direct-collocation boundary-value solve:

- Decision variables `X` (`NX × (Np+1)`), `U` (`1 × Np`), same shape
  convention as `mpc.py`.
- Cost: pure control effort, `sum(u_k²)` — no state tracking cost at all,
  since the endpoints are hard constraints rather than a target to trade
  off against other costs.
- Same per-step constraints as `mpc.py`'s NLP: RK4 dynamics consistency
  (`_rk4_step`, imported from `mpc.py`), cart acceleration bound
  (`sddot_max`), and the same speed-derated force bound with the same
  `fmax(0, ...)` floor and the same reasoning (an intermediate IPOPT
  iterate can transiently violate the velocity bound, which would
  otherwise make the force-bound interval empty).
- **Boundary constraints**: `X[:,0] == x0` and `X[:,Np] == xf` exactly
  (not penalized — hard equality), plus the usual rail (`s_max`) and
  velocity (`MOTOR_FREE_SPEED`) bounds over the whole trajectory.
- **Warm start**: `X_guess = linspace(x0, xf, Np+1)` — the naive,
  dynamics-ignorant straight-line guess described above. `U` initialized
  to all zeros.
- Solver options: same IPOPT `acceptable_*` early-certification pattern as
  `mpc.py`, with the comment noting the smoothed-Coulomb-friction term
  makes the very last digits of KKT convergence expensive, and a solution
  that's "optimal to acceptable level" for several iterations in a row is
  more than good enough to serve as a TVLQR reference. Tolerance kept at
  `1e-6` (not looser) so the trajectory stays dynamically consistent.
- On success, returns `(sol.value(X), sol.value(U), True)`. On solver
  failure (`RuntimeError`), returns the solver's last debug iterate
  instead (`opti.debug.value(...)`) with `success=False`, so a caller can
  still inspect how far it got.

### `_linearize_fns(params, dt) -> (A_fun, B_fun)` (private)

Builds CasADi `Function`s for the Jacobians of one RK4 step
(`_rk4_step`, from `mpc.py`) with respect to state and control —
`A = ∂x_next/∂x`, `B = ∂x_next/∂u` — used as the linearization primitives
for TVLQR below. Symbolic-differentiation-based, not finite differences.

### `tvlqr_gains(params, X_ref, U_ref, dt, Q, R, Qf) -> Ks (list of gain matrices)`

Backward Riccati recursion producing a **time-varying** LQR feedback gain
`K_k` at every step along the reference trajectory. The docstring explains
why this is necessary rather than just replaying `U_ref` open-loop: the
reference trajectory passes directly through the unstable inverted
equilibrium, so open-loop replay diverges — tiny numerical noise gets
chaotically amplified with nothing to correct it.

- For each step `k` from the end backward: linearizes at `(X_ref[:,k],
  U_ref[k])` to get `A`, `B`; computes the standard discrete-time LQR
  backward recursion `K = (R + B'SB)^-1 B'SA`, then updates the cost-to-go
  `S = Q + A'SA - A'SB K`.
- Returns the list of `K` gains, one per step, ordered forward-in-time (the
  loop fills `Ks[k]` at index `k` despite iterating in reverse).
- The docstring notes this linearization is only valid because `mpc.py`'s
  `_rk4_step` and `dynamics.py`'s plant are kept bit-identical (stated as
  verified: RK4 step diff == 0) — if the two ever drift apart, the
  computed gains would be for the wrong system.

### `track_trajectory(params, X_ref, U_ref, dt, s_max=0.18, Q=None, R=None, Qf=None) -> X_track (6, Np+1)`

Closed-loop tracking of the reference:

- Default `Q = diag([10, 50, 50, 1, 5, 5])`, `R = [[0.01]]`, `Qf = 10*Q` —
  a separate, generally looser weighting than `mpc.py`'s own defaults,
  tuned for tracking rather than regulation-from-a-perturbation.
- Computes `Ks` via `tvlqr_gains`.
- Defines a local `rk4` closure over `dynamics.py`'s `forward_dynamics`
  (yet another copy of the same RK4 pattern, here operating on the
  **actual plant**, not the CasADi model used to generate the reference).
- Loop over each reference step: `u = U_ref[k] - K_k @ (state -
  X_ref[:,k])` (feedforward + LQR correction), clipped to `±MOTOR_FORCE_MAX`,
  then steps the real plant one `dt`. The docstring frames this as
  feedforward doing the bulk of the work, with the LQR term correcting
  small deviations before they compound through the unstable region near
  upright.
- Returns the full closed-loop state history as a `(6, Np_ref+1)` array.

### `if __name__ == "__main__":` block

Solves a swing-up trajectory from hanging-down to upright via
`solve_swingup_trajectory`, prints solve success / endpoint reached / peak
`|s|` and `|u|`, then tracks it in closed loop via `track_trajectory`
against the real plant, printing final tracking error against the upright
target and a sparse side-by-side table of reference vs. tracked `s`/`th1`.

---

## 8. `visualize.py` — animation

**Purpose**: renders a swing-up run as a side-view Matplotlib animation —
cart on the rail, two pendulum links. Explicitly built on **`trajectory.py`**
(direct-collocation + TVLQR), not `swingup.py`'s energy-shaping heuristic,
because (per the module docstring) the heuristic plateaus short of upright
while the collocation approach actually converges — see `trajectory.py`'s
docstring for the underlying reason. `theta` is measured from vertical-up,
so a link tip is drawn at `pivot + l*(sin(theta), cos(theta))`.

### Module constant: `CART_W, CART_H = 0.08, 0.05`

Cart footprint in meters, assumed ~8 cm (MGN12 linear-rail carriage + motor
mount) — a placeholder to be corrected once the real carriage width is
measured.

### `animate(log, params, s_max=0.18, track_max=0.25, dt=0.05, save_path=None, fps=None) -> (fig, anim)`

- `log` is expected to be a sequence of rows shaped like
  `(t, s, th1, th2, sdot, th1dot, th2dot, u[, mode])` — note the indexing
  used (`row[1]`, `row[2]`, `row[3]`, `row[7]`) matches `trajectory.py`'s
  demo log format (index 0 is time), **not** `sim_loop.py`'s or
  `swingup.py`'s raw log tuples, which are shaped differently (`sim_loop.py`
  has no leading time column at index 0 the same way). The `__main__` block
  below builds the log in the exact shape `animate` expects.
- Sets up axes sized to the pendulum's full reach (`l1+l2+0.05`) vertically
  and `track_max` horizontally, draws the rail limits (`s_max`, dashed red)
  and the physical track ends (`track_max`, solid gray).
- Draws the cart as a `Rectangle` patch, and each link as a 2-point line
  (`link1_line`, `link2_line`) with markers at both ends (`"o-"` style).
- `joints(i)` (local closure): computes the three joint positions (cart
  pivot, link1 tip / link2 pivot, link2 tip) for frame `i` from
  `S[i], TH1[i], TH2[i]`.
- `init()` — FuncAnimation init callback, clears the line data and info
  text.
- `update(i)` — FuncAnimation per-frame callback: repositions the cart
  rectangle and both link lines via `joints(i)`, and updates a monospace
  text overlay showing `t, s, th1, th2, u`, plus a `[mode]` tag if the log
  row has more than 8 elements (i.e., only when `swingup.py`-style logs
  with a mode field are passed in — the demo below doesn't include one, so
  in practice this tag only appears if a caller passes a `swingup.py`-style
  log through `animate` directly).
- Builds a `matplotlib.animation.FuncAnimation` with `interval=dt*1000`ms
  per frame.
- If `save_path` is given, writes an animated GIF via
  `animation.PillowWriter` (works headless, no display needed), at `fps`
  (defaults to `round(1/dt)` if not given).
- Returns `(fig, anim)`.

### `if __name__ == "__main__":` block

CLI via `argparse`:
- `--save PATH` — also write an animated GIF to `PATH`.
- `--no-show` — skip opening a live interactive window (for headless runs).

Body: solves a swing-up trajectory with `solve_swingup_trajectory`, tracks
it with `track_trajectory` against the real plant, reassembles the tracked
states plus the reference control sequence into the `(t, s, th1, th2, sdot,
th1dot, th2dot, u)` row format `animate` expects, calls `animate(...)`, and
`plt.show()`s it unless `--no-show` was passed.

---

## 9. How the files relate to each other

```
derive_eom.py  ─┐ (symbolic cross-check only, not imported by anything)
                 ╲
dynamics.py ──────┼──── forward_dynamics(): the plant, used by:
   │               │       sim_loop.py, swingup.py, trajectory.py (track_trajectory)
   │
mpc.py ───────────┼──── _dynamics()/_rk4_step(): CasADi model, reused (imported) by:
   │  MPCController        trajectory.py (both functions), swingup.py (constants only)
   │  MOTOR_FORCE_MAX,
   │  MOTOR_FREE_SPEED  ── shared force/speed envelope constants, imported by:
   │                        sim_loop.py, swingup.py, trajectory.py
   │
   ├── sim_loop.py: MPCController + dynamics.py, closed-loop regulation demo
   │
   ├── swingup.py: reactive energy-shaping swing-up (own control law) → 
   │               hands off to MPCController for catch/stabilize
   │
   └── trajectory.py: offline collocation swing-up (reuses mpc.py's _dynamics/
                       _rk4_step directly) → TVLQR tracking against dynamics.py
                       │
                       └── visualize.py: animates a trajectory.py run
```

**Cross-file invariants worth knowing if you touch any of this code**:

- `FRICTION_EPS = 0.05` is duplicated in `dynamics.py` and `mpc.py` and
  must stay equal — it's not imported from one to the other.
- The 3×3 mass-matrix formula (`M11..M33`) is implemented **three times**:
  `dynamics.py` (torch), `mpc.py._dynamics` (CasADi), and
  `swingup.py._mass_matrix` (numpy). All three must stay in exact agreement
  or the different controllers are literally modeling different plants.
- `mpc.py`'s `MOTOR_FORCE_MAX` / `MOTOR_FREE_SPEED` are the single source
  of truth for the actuator envelope and are imported (not recomputed)
  everywhere else.
- Every file's RK4 integrator is a separate ~4-line implementation
  (`mpc._rk4_step` for CasADi, plus one ad hoc torch version each in
  `sim_loop.py`, `swingup.py`, and `trajectory.py.track_trajectory`) — not
  factored into a shared helper.
- All physical-parameter dicts across every file's `__main__` demo use the
  identical example configuration (`m1=0.12, m2=0.09, l1=0.3, l2=0.25,
  M=1.0, m_enc1=0.18, m_enc2=0.18`, uniform-rod inertias), and all demos
  start from the same hanging-down initial state `[0, π, π, 0, 0, 0]`.

## 10. Dependencies (from `requirements.txt`)

- `torch` — `dynamics.py`'s plant and every RK4 integrator built on it.
- `casadi` (pinned `3.7.2`) — `mpc.py` and `trajectory.py`'s NLPs, solved
  with the bundled IPOPT.
- `numpy` (pinned `1.24.3`) — array plumbing throughout; pinned because
  newer `scipy` would force a `numpy` version incompatible with this
  `casadi` build.
- `scipy` (pinned `1.10.1`) — not used directly by the files above; used
  elsewhere in the project (QMC sampling, `solve_discrete_are` for LQR).
- `matplotlib` — `visualize.py`'s animation.
- `sympy` — `derive_eom.py`'s symbolic derivation (not listed in
  `requirements.txt`; only needed if you run that file directly).
