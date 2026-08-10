# Speaking Outline — PINN Project (Özyeğin Summer Research, 5 min)

**The story in one line:** I built a physics-informed controller for a double inverted pendulum, and when I evaluated it closed-loop the simulation told me the *hardware* was the problem — before the rig was finished.

**Delivery:** Talk to the audience, not the slides. State each slide's printed question, then answer it. On slide 4, do not soften the failure — the failure *is* the result.

| Slide | Section | Budget |
|-------|---------|--------|
| 1 | Title | ~20 s |
| 2 | 01 · Theory | ~55 s |
| 3 | 02 Simulation / 03 Application | ~60 s |
| 4 | 04 · Prototype & Results | ~110 s |
| 5 | Closing | ~45 s |

⚠️ **Read fully, slide 4 runs ~130 s and the whole talk ~5:45.** Two marked `[CUT]` blocks on slide 4 bring it to ~5:00. Practise it once out loud with a timer — if you hit slide 4 later than the 2:15 mark, take both cuts.

---

## Slide 1 — Title (~25 s)
- "I'm Çağlar Ayyıldız; my mentor is Burçin Ünlü."
- One-liner — **note the verb: "learn to balance", not "balance". It does not balance yet, and slide 4 says so; don't contradict yourself in your first sentence:** *"I built a pipeline that trains one neural network to learn balancing control for a double inverted pendulum across many pendulum sizes — and along the way the simulation caught a hardware design flaw before I finished building the rig."*
- **Transition:** "First, what I'm actually controlling."

## Slide 2 — 01 · Theory (~65 s)
> *What phenomenon are you studying, and what question are you asking?*
- **Phenomenon:** a cart carrying two stacked pendulums — unstable and *underactuated*: one motor, three things to control.
- **Equation** — don't read symbols, say the meaning: *"This is F = ma for the whole system. On the left: inertia, the forces from its own swinging, and gravity — which for an inverted pendulum actively tips it over. On the right: my one motor."*
- Point at **B = [1,0,0]ᵀ**: "the motor only touches the cart. I can't grab the links — that's what underactuated means, and that's the hard part."
- **Research question (read it):** *"Can one network, given the pendulum's physical parameters as input, learn an MPC-equivalent controller that works on pendulum shapes it never trained on?"*
- **Prior work (one sentence):** "MPC solves this but must be re-solved per pendulum; DAgger distills a controller into a fast network; PINNs bake the equations into training. I combine all three."
- **Transition:** "Because that equation is differentiable, it becomes the simulator everything else is built on."

## Slide 3 — 02 Simulation / 03 Application (~75 s)
> *How did you model it, and what did you build?*
- **Simulation:** "A differentiable PyTorch model of the dynamics, plus a CasADi MPC controller as the *teacher* — it solves the optimal command for hundreds of random pendulums, and that becomes labeled data."
- **Four training signals:** "Imitate MPC; roll the network's *own* commands through the real dynamics and penalize drifting from upright; penalize breaking the limits; and — on random pendulums the teacher never solved — penalize commands that push a link the wrong way. Weighting anneals from data-heavy to physics-heavy."
- **DAgger in one line:** "A copied controller drifts into states the teacher never showed it. So I let the network drive, find where it drifts, ask MPC what to do *there*, and retrain."
- **Application:** "State plus the four physical parameters → one voltage, single forward pass. About 26 thousand multiply-adds, sub-millisecond — that's why it fits a microcontroller and MPC doesn't. Swap the parameters, control a different pendulum, no retraining."
- **Assumptions, say them:** "MPC is treated as ground truth, friction is idealized, and I stay near upright — not swing-up."
- **Transition:** "Then I evaluated it closed-loop, and everything failed."

## Slide 4 — 04 · Prototype & Results (~85 s) — **the core of the talk**
> *What did you build or measure, and what did you find?*
- **Open with the failure, flatly:** "Every controller failed to balance in closed-loop simulation. Not just my network — the MPC teacher failed too, and so did an LQR baseline."
- **The reasoning, step by step** (this is what earns credit):
  - "When your teacher *and* your baseline both fail, the problem usually isn't the student. So I checked the obvious suspects and ruled them out."
  - ⏱️ **[CUT #1 if running long — these two lines are the most droppable]**
  - "Not the timestep — LQR died at the same physical time whether I ran at 20 Hz or 200 Hz."
  - "Not a broken gain — the LQR is provably stable on the linearization, max eigenvalue 0.93, and the system is fully controllable."
  - "Then I looked at the trajectories: **cart velocity pinned at 0.3996 metres per second in every failed run.** The motor's ceiling is 0.4. The cart was flat out and still couldn't get back under the pendulum."
- **The sweep (point at the chart)** — **say "the teacher", never a bare "it": the chart is MPC, not your network:** "So I swept just the speed ceiling, with the MPC teacher driving. At 0.4 m/s the teacher can't balance at all. At 1.2 m/s it can. Nothing else changed."
- **The cause (point at the table):** "The gearing. A 20-tooth pulley gives 173 newtons and 0.4 m/s. But 173 newtons accelerates a 1.5 kg cart at 115 m/s² — absurd overkill — while the speed starves. A 60-tooth pulley trades that unused force for 1.2 m/s. It costs nothing I was using."
- **Land the point:** "So the deliverable isn't just a controller — the simulation caught an infeasible hardware spec *before* the rig was built. That's worth more than a working demo would have been."
- **Then the honest half — do not skip this, say it plainly:** "I regenerated the whole dataset at the corrected spec — 200 out of 200 configs now solve, zero dropped, where the old spec was rejecting labels. And the learned controller *still* doesn't balance: 0% over 64 rollouts, against 12% for LQR. It holds the links near upright, peaking around 0.28 radians, and then drifts off the rail."
- **The diagnostic lead (this is the part that shows judgement — keep it, cut elsewhere):** "I ran a DAgger round on top. It cut validation error by 31% and changed the closed-loop behaviour by nothing. So the bottleneck is *not* imitation quality — most likely cart-position regulation is under-weighted against angle in the teacher's cost. That's my next experiment, not a mystery."
- ⏱️ **[CUT #2 if running long: drop the "200 out of 200 configs, zero dropped" clause from the line above — it's supporting detail, not the point.]**
- **Transition:** "Which leaves an honest scorecard."

## Slide 5 — Closing (~50 s)
- **Found:** "The pipeline runs end-to-end, and its most useful output was a hardware verdict: the original gearing can't balance this pendulum; 60-tooth can. Every config solves at the corrected spec."
- **Limitation — own it, don't hedge:** "The learned controller does not balance yet — LQR beats it. I have not proven the physics-informed version beats plain imitation, extrapolation is untested, and nothing has run on the physical rig."
- **Next:** "Chase the drift — I think it's the cost weighting on cart position. Then ablation and generalization at the corrected spec, then hardware: system-ID, encoder filtering, weight export, safety watchdog."
- "Thank you — happy to take questions."

---

## Q&A prep

**On the finding**
- *"Isn't a failed controller a failed project?"* → The controller was never the bottleneck — the plant was infeasible as specified. Finding that in simulation, before committing the mechanics, is exactly what the simulation was for. Now the spec is corrected and the same pipeline runs on it.
- *"How do you know it's speed and not force?"* → Three ways: velocity pinned at exactly the motor ceiling in every failure; sweeping only the speed limit changes the outcome; and the *revised* gearing has 3× **less** force yet lets the teacher balance.
- *"Why did the MPC teacher fail if it generated good training data?"* → Each label is a single open-loop solve over a 20-step horizon, which stays feasible. Iterating it closed-loop is what runs the cart into the speed limit. The labels were fine; the closed loop wasn't.
- *"Why does LQR beat your PINN?"* → It does, 12% to 0%, and I'm not going to dress that up. LQR is an optimal controller for the linearization, and near upright that's a strong baseline. My network imitates a teacher whose closed-loop cost apparently tolerates cart drift, and it inherits that. Fixing the teacher's cost weighting is the next step, not adding network capacity.
- *"Did DAgger help?"* → On validation, yes — 31% lower. On closed-loop balance, not at all. That's informative: it rules out imitation accuracy as the bottleneck and points at the objective itself.

**On the method**
- *"What makes this a PINN, not plain imitation?"* → Three physics losses on top of imitation: a differentiable rollout penalizing drift from upright, a barrier on limits, and a collocation term on shapes the teacher never solved. Honest caveat: it's PINN-*inspired* — Raissi's mechanism (differentiable physics + collocation in the loss) applied as a control-stability objective, not a literal PDE residual.
- *"Why output voltage, not force?"* → 24 V bus; `tanh` hard-bounds the command to ±24 V by construction — a structural safety limit no soft penalty can guarantee.
- *"Real-time feasible?"* → ~26k multiply-adds, sub-millisecond, about 1% of a 20 Hz loop on an ESP32.
- *"Which checkpoint are these numbers from?"* → Be precise: the earlier 0.66 validation figure came from an epoch-26 checkpoint saved during the data-only warmup — that's a plain imitation net, not the PINN. Don't quote it as a PINN result.

**On hardware**
- *"Is it running on the rig?"* → No. The rig is built, but weight export, encoder filtering, and system-ID are not done. Saying otherwise would be dishonest.
- *"Is a stepper the right actuator?"* → Open question. A stepper driven open-loop is a position device, not a clean force source, so the voltage→force map is provisional until system-ID.
- *"Where do the motor numbers come from?"* → Catalog specs for the 57HS82 (2.2 N·m holding, 600 RPM ceiling with a common driver), with a 0.5 dynamic derate — not measured on my unit. That's flagged as a thing to verify.

---
*Fill before presenting: presentation **date** (slide 1); confirm the **reference list**.*
