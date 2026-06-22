# Roybot — Drive + Chase Core (Sub-project #1) — Design Spec

**Status:** Approved (design phase) · **Date:** 2026-06-22 · **Owner:** becqu

A small, fast, durable, autonomous robot that plays with a cat (Roy). It is a wheeled
re-imagining of [GrowBot V0](https://github.com/britcruise9/GrowBot) (Art of the Problem /
Brit Cruise): reuse GrowBot's driver/IO scaffolding and "LLM-goals + fast-local-reflexes"
architecture, but swap the learned-legged-locomotion for a **differential-drive base** and
point the same MuJoCo→RL→deploy pipeline at the genuinely hard, fun problem: **learning to
chase and play**.

This spec covers **sub-project #1 only**: the drive base, the closed-form driver, the MuJoCo
digital twin + simulated cat, the learned chase/play policy, and the onboard runtime.
Perception/brain (#2) and the self-charging dock (#3) get their own specs.

---

## 1. Goal & context

- **Purpose:** a "sentient"-feeling autonomous toy that engages Roy in active play (chasing a
  captive lure, darting/juking) while the owner is home, and self-docks to recharge when low.
- **Roy:** "normal play" — pounces, bunny-kicks, occasionally rough. Sets a moderate
  durability bar (tough shell, can't-flip-or-self-rights, no parts to strip/swallow).
- **Whole-robot vision (3 sub-projects):**
  1. **Drive + Chase core** *(this spec)* — move, and learn to chase/play.
  2. **Perception + brain** — vision finds/tracks Roy; cloud LLM sets high-level goals/personality.
  3. **Self-charging dock** — battery-low → IR homing → mechanical funnel → pogo-pin charge.
- **Knowledge foundation:** karpathy-guidelines (surface assumptions, surgical, verifiable
  success criteria) + ponytail (laziest thing that works) govern *how* we build. GrowBot is the
  *what* we adapt.

## 2. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Drivetrain | **Differential / skid-steer**, 2× DC gearmotors (N20-class + encoders) + dual H-bridge | Turn-in-place agility is ideal for juking a cat; fewest parts; most durable; smallest/fastest |
| Locomotion control | **Closed-form** `(v_fwd, v_yaw) → wheel speeds` + per-motor calibration | Wheel kinematics are solved; learning to move would be over-engineering |
| Learned policy | **Chase/play policy** — RL in MuJoCo vs a simulated moving "cat" | Same training structure the owner wanted, aimed at the part that's actually hard |
| RL inputs | **Abstract cat-position vector** (not pixels) | Keeps the RL problem tiny — no CNN, trains on a 3070 in minutes |
| Compute | **Pi Zero 2 W** onboard; tiny MLP policy @50 Hz; cloud LLM for high-level goals | Same as GrowBot; Pi runs the policy easily, only vision strains it |
| Training compute | **RTX 3070 + MJX/Brax** (SB3+MuJoCo CPU fallback). **No H100 needed** | Problem is tiny; H100 only justified if we ever train from pixels (we don't) |
| Power | Light (2 DC motors) — small **1S or 2S LiPo**, finalized in Phase 2 | 2 motors draw far less than the 4-servo legged plan; relaxes the heavy 2S/UBEC requirement |
| Charging | **Pogo-pin contacts + passive funnel + IR homing** (not Qi) | Qi tolerates ~±5 mm; a legged/wheeled robot docks with ~20 mm error → Qi would often not charge |
| Brain | Cloud LLM picks goals → chase policy + local reflexes execute | Pi Zero's 512 MB can't host an LLM; fast motion stays local |
| Safety/durability | Captive chew-proof lure, shrouded wheels/gaps, no swallowable parts, PETG, low-CG/self-righting, stall cutoff | "Normal play" bar; cat-safety is non-negotiable |

## 3. Reuse map (from GrowBot V0)

| GrowBot asset | Verdict |
|---|---|
| `imu.py` (MPU-6050) | **Reuse as-is** — proprioception for the policy + tip/stall reflexes |
| `leds.py` / `audio.py` (WS2812 ring, MAX98357A) | **Reuse** — expressive "personality" output |
| `camera.py` (OV5647) | **Extend** later (#2) — frame stream for cat-tracking |
| `pins.py` | **Edit** — drop servo bus; add motor-driver + encoder pins |
| `servos.py` (SCS0009 serial bus) | **Drop** — replaced by DC-motor + H-bridge control |
| `simulation/…body.xml` (2-leg MuJoCo) | **Replace** — new wheeled differential-drive twin |
| `requirements.txt` | Base reuse; training deps (mujoco, sb3/brax) live on the dev PC, not the Pi |

> GrowBot's training pipeline, reward functions, learned policies, and LLM agent loop are
> **not released** ("V1, Fall 2026"). We build our own. We do not wait for V1.

## 4. Design: the Drive + Chase core

### A. Drive base (hardware)
- Small PETG chassis sized around motors + battery. **2× N20 gearmotors + magnetic encoders**,
  **dual H-bridge (TB6612FNG)** preferred over DRV8833 (higher current/efficiency), skid-steer.
- Wheels: **2 drive wheels + 1 caster/ball** (hard floors) *or* **4-wheel tank** (carpet grip)
  — settled in Phase 2 by Roy's actual floors.
- Motors run off battery voltage via the H-bridge; the Pi runs off a dedicated 5 V regulator
  (separate from the motor rail) + bulk cap so motor surges can't brown out the Pi.
- **Low CG so it cannot easily flip**, or shaped to self-right. Encoders → closed-loop wheel
  speed + odometry (also helps the dock approach in #3).

### B. Closed-form driver (`drive.py`)
- One small module: `drive(v_fwd, v_yaw) -> (pwm_left, pwm_right)` with per-motor calibration
  (real motors differ in gain/deadband — calibration knob stays in, per ponytail's hardware rule).
- Optional encoder-based closed-loop speed control (PI) if open-loop tracking is too sloppy.
- Replaces GrowBot's entire servo layer. **One runnable self-check** (assert the mapping:
  straight = equal PWM, spin = opposite PWM, scaling monotonic).

### C. Digital twin (MuJoCo MJCF) + simulated cat
- New wheeled differential-drive model: chassis body, 2 driven hinge wheels (velocity/torque
  actuators), caster, realistic mass/inertia/friction, **IMU site** (accel+gyro), modeled after
  GrowBot's measured-geometry approach.
- **Simulated "cat":** a moving target with stochastic flee/dart/idle behavior (scripted
  prey, randomized speed/heading/pauses) for the policy to chase. Self-play co-evolution is
  explicitly **out of scope** (YAGNI) — a scripted stochastic prey is enough; revisit only if
  play looks too easy.
- (A pure-2D kinematic sim would be lazier, but MuJoCo buys tipping dynamics, IMU, and a real
  sim2real path — worth the small extra cost.)

### D. Chase/play policy (the RL centerpiece)
- **Observation (proprioception-light, ~10–15 dims):** relative cat position + velocity, robot
  linear/angular velocity, IMU roll/pitch + rates, last action. *Not* raw motor state.
- **Action (2 dims):** velocity command `(v_fwd, v_yaw)` — routed *through* the closed-form
  driver so the net never re-learns trivial motor math.
- **Reward (engaging play, legged_gym-style shaping):** stay in an interesting band near the
  cat (not too far, not ramming), reward darts/juking and distance variation (keep-away
  dynamics), penalize tipping, excessive energy, and jerky action. A tunable "playfulness"
  weight.
- **Training backend:** start with **stable-baselines3 + MuJoCo** (simplest working loop), then
  **MJX/Brax PPO on the RTX 3070** for fast iteration. **No H100** — the abstract-obs, tiny-MLP
  problem trains in seconds-to-minutes on the 3070; an H100 only pays off for pixels-in training,
  which we don't do.
- **Domain randomization:** mass ±20%, friction, control latency 0–40 ms, per-motor gain/offset
  /deadband, sim cat behavior. Train off-board; deploy a tiny MLP.

### E. Onboard runtime + reflexes
- Policy runs **@50 Hz** on the Pi from live inputs (cat-position from #2's perception + IMU);
  sub-ms inference (Pi is not the bottleneck).
- **Safety floor underneath the policy (always on):** stall detect (encoder vs command) → cut
  motors; tip detect (IMU) → stop/self-right; bump/edge → back off. Calibration knobs left in.

## 5. BOM deltas (vs GrowBot V0)

Only the drive+chase-core deltas are listed; dock (#3) and perception (#2) parts are speced there.

| Change | Part (indicative) | Note |
|---|---|---|
| **Remove** | 2× SCS0009 servos, 1 kΩ resistor, MT3608 boost | Servo bus replaced |
| **Add** | 2× N20 gearmotor **with encoder** | Pick gear ratio for speed/torque balance |
| **Add** | Dual H-bridge **TB6612FNG** | Higher efficiency than DRV8833 at this scale |
| **Add** | 2 drive wheels (+ caster) or 4-wheel set | Hard-floor vs carpet — set in Phase 2 |
| **Add** | Captive lure: feather/pom on a short rigid/braided arm, no free end | **#1 cat safety item** |
| **Keep** | Pi Zero 2 W, MPU-6050, OV5647 cam, WS2812 ring, MAX98357A + speaker | From GrowBot |
| **Revisit** | Battery (1S or 2S LiPo) + 5 V regulator for Pi + charge mgmt | Lighter than legged plan; finalize cell count w/ a quick power budget |

## 6. Cat-safety & durability rules (non-negotiable)

- **Lure = #1 hazard (linear foreign body).** No loose string/yarn (can saw through a cat's gut,
  often fatal; cats can't spit it out). Use a **captive, chew-proof lure** (feather/pom on a
  short ~10–20 cm rigid/braided arm, no free end, internally over-molded). **Attended-play only;
  store after use.** Owner note: if string is visible from mouth/rear or ingestion is suspected,
  vet **same-day — do not pull**.
- **Pinch/entanglement (ISO 13854):** keep moving-part gaps **<4–6 mm or >25 mm** (never the
  4–25 mm trap band); shroud wheels and any moving gaps; round all edges (claws/whiskers/tail).
- **Choking (ASTM F963):** no detachable part may fit in the 31.7 × 57.1 mm small-parts cylinder
  — captive/recessed/glued fasteners, tool-locked battery hatch, molded-in features.
- **Materials:** PETG shell (tougher than brittle PLA), known-safe filament, integral color or
  fully-cured non-toxic finish (no flaking paint).
- **Durability:** stall cutoff on motors; low-CG ballasted shell for no-flip / passive
  self-right; thick walls at impact corners; compliant-mounted PCB/battery.

## 7. Training stack & sim2real notes

- **Pipeline:** author wheeled MJCF → train PPO → export tiny MLP (2×[128–256]) to ONNX/NumPy →
  run @50 Hz on the Pi.
- **Backends:** SB3 + MuJoCo (CPU-capable, simplest) → MJX/Brax PPO on the 3070 (minute-scale).
  SBX is a faster middle option. Skip IsaacLab.
- **Sim2real (from Tan et al. 2018 / Minitaur):** model motors as their real transfer function
  (PWM→speed with deadband + saturation), inject 10–30 ms latency, proprioception-light obs,
  domain-randomize. Closest DIY reference: **Stanford Pupper v3** (Pi + serial actuators +
  MuJoCo + RL→real).
- **Brain interface (#2):** the LLM emits high-level goals ("find Roy", "play", "rest", "go
  dock") that set the chase policy's mode/target — never raw motor commands.

## 8. Build phases & realistic timing

- **Phase 1 — pure software, start immediately, zero hardware:** wheeled twin + closed-form
  driver (sim) + simulated cat + trained chase/play policy, validated in a sim demo of Roybot
  chasing/playing without tipping. **Coding: a focused session or two** on the 3070. No waiting.
- **Phase 2 — hardware, calendar-bound:** order BOM (motors/driver/battery — **shipping is the
  real wait, ~1–3 weeks**), print chassis, wire, bring up the driver, deploy the policy,
  sim2real-tune, real chase. Then perception (#2) feeds real cat-position in.

## 9. Success criteria (verifiable)

1. Wheeled twin loads in MuJoCo; `drive()` moves/turns it in sim.
2. Simulated cat moves/flees/darts stochastically.
3. Trained policy chases + juke-plays around the sim cat **without tipping** (meets a defined
   "time-in-play-band" + "distance-variation" threshold).
4. Policy exports and runs **@50 Hz on the Pi**.
5. Real base drives forward and **turns in place**.
6. Real base executes the policy's velocity commands.
7. Safety floor cuts motors on stall/tip.

## 10. Open items (settle later, not blocking Phase 1)

- 2-wheel+caster vs 4-wheel tank, wheel type/diameter, top speed → by Roy's floors (Phase 2).
- Exact battery cell count + motor voltage + 5 V regulator → quick power budget (Phase 2);
  couples to the dock charge path (#3: 1S TP4056 vs 2S BMS).
- "Playfulness" reward weights → tuned empirically in sim then on Roy.
- Perception approach (on-Pi tiny detector vs offloaded) → sub-project #2.

## 11. References

- GrowBot V0 — https://github.com/britcruise9/GrowBot (drivers, BOM, MuJoCo body, build video)
- Tan et al. 2018, "Sim-to-Real: Learning Agile Locomotion for Quadruped Robots"
- legged_gym (ETH RSL / Rudin et al.) — reward-term structure reference
- stable-baselines3 (PPO) · MuJoCo MJX / MuJoCo Playground (Brax PPO)
- Stanford Pupper v3 — closest DIY Pi + sim + RL→real analog
- Motor/electronics: N20 gearmotor + encoder, TB6612FNG dual H-bridge, TSOP38238 IR receiver
  (dock #3), sprung pogo-pin connectors (dock #3)
