# Roybot — Drive + Chase Core (Sub-project #1) — Design Spec

**Status:** Approved (design phase) · **Date:** 2026-06-22 · **Owner:** becqu

A small, fast, durable, autonomous robot that plays with a cat (Roy) and grows into a curious
companion. It is a wheeled re-imagining of [GrowBot V0](https://github.com/britcruise9/GrowBot)
(Art of the Problem / Brit Cruise): reuse GrowBot's driver/IO scaffolding and its
"fast-local-autonomy + offboard-reflection" spirit, but swap learned-legged-locomotion for a
**differential-drive base** and point the same MuJoCo→RL→deploy pipeline at the genuinely hard,
fun problem: **learning to play with Roy on her terms.**

The full companion vision (memory, curiosity, dreaming, owner reports) lives in the
[vision & roadmap doc](2026-06-22-roybot-vision-and-roadmap.md). **This spec covers
sub-project #1 only:** the drive base, the closed-form driver, the MuJoCo digital twin +
simulated cat, the learned **consent-based** chase/play policy, and the onboard runtime.

---

## 1. Goal & context

- **Purpose:** an autonomous toy-growing-into-companion that engages Roy in active play (chasing
  a captive lure, darting/juking) **only when she wants it**, and self-docks to recharge.
- **Roy:** "normal play" — pounces, bunny-kicks, occasionally rough. Moderate durability bar.
- **Play ethic (first-class requirement):** *consent-based play.* Roybot must never pester —
  it backs off and gives space when Roy is disinterested. In sub-project #1 this is **learned**
  in the chase reward, not bolted on.
- **Foundation:** karpathy-guidelines (surface assumptions, surgical, verifiable criteria) +
  ponytail (laziest thing that works) govern *how*; GrowBot is the *what* we adapt.

## 2. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Drivetrain | **Differential / skid-steer**, 2× N20 gearmotors (+encoders) + dual H-bridge | Turn-in-place is ideal for juking a cat; fewest parts; durable; small/fast |
| Locomotion control | **Closed-form** `(v_fwd, v_yaw) → wheel speeds` + per-motor calibration | Wheel kinematics are solved; learning to move would be over-engineering |
| Learned policy | **Consent-based chase/play policy** — RL in MuJoCo vs a moody simulated "cat" | Same training structure wanted, aimed at the part that's actually hard |
| RL inputs | **Abstract cat position + engagement** (not pixels) | Keeps the RL problem tiny — trains on a 3070 in minutes |
| Onboard autonomy | **Local behavior + reflexes on the Pi, no LLM in the loop** | Play/explore/leave-alone must be instant, private, offline |
| LLM (language only) | **Cloud now** — OpenRouter / DeepSeek V4 Flash (swappable) → **local Spark home-base (~1 yr)** | A cat toy doesn't need an LLM; the LLM is for reports/talk/dream-narration. Provider behind one interface |
| Compute | **Pi Zero 2 W** onboard; tiny MLP policy @50 Hz | Same as GrowBot; Pi runs the policy easily |
| Training/dream compute | **becqu's RTX 3070** (MJX/Brax; SB3+MuJoCo fallback). **No H100** | Tiny abstract-obs problem; H100 only for pixels-in (not done) |
| Home base / storage | **becqu's PC (RTX 3070)** — recordings + nightly "dream" (consolidate logs, learn Roy prefs, fine-tune policy) | Native local storage; GPU for retraining |
| Power | Light (2 DC motors) — small **1S or 2S LiPo**, finalized in Phase 2 | 2 motors draw far less than the 4-servo legged plan |
| Charging | **Pogo-pin + passive funnel + IR homing** (not Qi) | Qi ~±5 mm vs ~20 mm real docking error → Qi would often not charge |
| Safety/durability | Captive chew-proof lure, shrouded wheels/gaps, no swallowable parts, PETG, low-CG/self-righting, stall cutoff | "Normal play" bar; cat-safety non-negotiable |

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
  **dual H-bridge (TB6612FNG)** (higher efficiency than DRV8833), skid-steer.
- Wheels: **2 drive wheels + 1 caster/ball** (hard floors) *or* **4-wheel tank** (carpet grip)
  — settled in Phase 2 by Roy's actual floors.
- Motors run off battery voltage via the H-bridge; the Pi runs off a dedicated 5 V regulator
  + bulk cap so motor surges can't brown out the Pi.
- **Low CG so it cannot easily flip**, or shaped to self-right. Encoders → closed-loop wheel
  speed + odometry (also helps the dock approach in #3).

### B. Closed-form driver (`drive.py`)
- One small module: `drive(v_fwd, v_yaw) -> (pwm_left, pwm_right)` with per-motor calibration
  (gain/deadband knobs stay in, per ponytail's hardware rule). Optional encoder PI loop if
  open-loop tracking is too sloppy. Replaces GrowBot's whole servo layer. **One runnable
  self-check** of the mapping (straight = equal, spin = opposite, scaling monotonic).

### C. Digital twin (MuJoCo MJCF) + a moody simulated cat
- Wheeled differential-drive model: chassis, 2 driven wheels (velocity/torque actuators),
  caster, realistic mass/inertia/friction, **IMU site**, modeled after GrowBot's measured-geometry
  approach.
- **Simulated "cat" with a mood/engagement state** (this is what makes consent-based play
  learnable):
  - Position dynamics: stochastic flee / dart / idle.
  - **Engagement** (0–1): rises with good play (robot in the interesting band, playful motion,
    appropriate spacing), **falls when over-chased/pestered** or naturally over time. Below a
    threshold the cat becomes *disinterested* — walks away and ignores the robot.
  - Randomized per-episode "personality" (some sessions more playful, some less).
- Self-play co-evolution is **out of scope** (YAGNI) — scripted moody prey is enough.

### D. Consent-based chase/play policy (the RL centerpiece)
- **Observation (~10–15 dims):** relative cat position + velocity, **cat engagement signal**,
  robot linear/angular velocity, IMU roll/pitch + rates, last action. *Not* raw motor state.
- **Action (2 dims):** velocity command `(v_fwd, v_yaw)` — routed *through* the closed-form
  driver so the net never re-learns trivial motor math.
- **Reward (consent-based play):**
  - **When the cat is willing:** reward engaging — staying in the interesting band, darts/juking,
    distance variation (keep-away, not ramming).
  - **When the cat is disinterested:** reward **giving space** — backing off, idling at a
    respectful distance, *not* approaching. **Penalize pestering** a disinterested cat.
  - Always: penalize tipping, excessive energy, jerky action.
  - Tunable "playfulness" and "politeness" weights.
- **Backend:** SB3 + MuJoCo first (simplest loop) → MJX/Brax PPO on the RTX 3070 (minute-scale).
  **No H100.** Domain-randomize mass/friction/latency/motor-gain + the cat's behavior/personality.
  Train off-board; deploy a tiny MLP.

### E. Onboard runtime + reflexes
- Policy runs **@50 Hz** on the Pi from live inputs (cat position + engagement from #2's
  perception, + IMU); sub-ms inference.
- **Safety floor (always on):** stall detect (encoder vs command) → cut motors; tip detect (IMU)
  → stop/self-right; bump/edge → back off. Calibration knobs left in.

## 5. BOM deltas (vs GrowBot V0)

Drive+chase-core deltas only; dock (#3), perception (#2), and companion (#5) parts are speced there.

| Change | Part (indicative) | Note |
|---|---|---|
| **Remove** | 2× SCS0009 servos, 1 kΩ resistor, MT3608 boost | Servo bus replaced |
| **Add** | 2× N20 gearmotor **with encoder** | Pick gear ratio for speed/torque balance |
| **Add** | Dual H-bridge **TB6612FNG** | Higher efficiency than DRV8833 at this scale |
| **Add** | 2 drive wheels (+ caster) or 4-wheel set | Hard-floor vs carpet — set in Phase 2 |
| **Add** | Captive lure: feather/pom on a short rigid/braided arm, no free end | **#1 cat safety item** |
| **Keep** | Pi Zero 2 W, MPU-6050, OV5647 cam, WS2812 ring, MAX98357A + speaker + INMP441 mic | From GrowBot (mic/cam used by #2/#5) |
| **Revisit** | Battery (1S or 2S LiPo) + 5 V regulator for Pi + charge mgmt | Lighter than legged plan; finalize cell count w/ a quick power budget |

## 6. Cat-safety & durability rules (non-negotiable)

- **Lure = #1 hazard (linear foreign body).** No loose string/yarn (can saw through a cat's gut,
  often fatal; cats can't spit it out). Use a **captive, chew-proof lure** (feather/pom on a
  short ~10–20 cm rigid/braided arm, no free end, internally over-molded). **Attended-play only;
  store after use.** If string is ever seen from mouth/rear or ingestion suspected: vet
  **same-day — do not pull**.
- **Pinch/entanglement (ISO 13854):** moving-part gaps **<4–6 mm or >25 mm** (never 4–25 mm);
  shroud wheels and moving gaps; round all edges.
- **Choking (ASTM F963):** no detachable part fits the 31.7 × 57.1 mm small-parts cylinder —
  captive/recessed/glued fasteners, tool-locked battery hatch, molded-in features.
- **Materials:** PETG shell, known-safe filament, integral color or fully-cured non-toxic finish.
- **Durability:** stall cutoff; low-CG ballasted shell for no-flip / passive self-right; thick
  walls at impact corners; compliant-mounted PCB/battery.

## 7. Training stack & sim2real notes

- **Pipeline:** wheeled MJCF → PPO → export tiny MLP (2×[128–256]) to ONNX/NumPy → @50 Hz on Pi.
- **Backends:** SB3 + MuJoCo (simplest) → MJX/Brax PPO on the 3070 (minute-scale). Skip IsaacLab.
- **Sim2real (Tan et al. 2018 / Minitaur):** model motors as their real transfer function
  (PWM→speed, deadband + saturation), inject 10–30 ms latency, proprioception-light obs,
  domain-randomize. Closest DIY reference: **Stanford Pupper v3**.
- **Interface to the brain (#2):** the behavior arbiter emits the policy's *mode/target*
  ("play toward Roy", "give space", "idle") — never raw motor commands.

## 8. Build phases & realistic timing

- **Phase 1 — pure software, start immediately, zero hardware:** wheeled twin + closed-form
  driver + moody simulated cat + trained **consent-based** chase policy, validated in a sim demo
  (Roybot plays when invited, gives space when not, never tips). **Coding: a focused session or
  two** on the 3070. No waiting.
- **Phase 2 — hardware, calendar-bound:** order BOM (motors/driver/battery — **shipping ~1–3
  weeks** is the real wait), print chassis, wire, bring up the driver, deploy the policy,
  sim2real-tune, real chase. Then perception (#2) feeds real cat position + engagement in.

## 9. Success criteria (verifiable)

1. Wheeled twin loads in MuJoCo; `drive()` moves/turns it in sim.
2. Simulated cat moves/flees/darts **and** its engagement rises/falls with the robot's behavior.
3. Trained policy chases + juke-plays around a *willing* sim cat **without tipping** (meets a
   defined "time-in-play-band" + "distance-variation" threshold).
4. **Consent test:** when the sim cat goes disinterested, the policy **backs off and gives space**
   (approach rate drops below a threshold — it does not pester).
5. Policy exports and runs **@50 Hz on the Pi**.
6. Real base drives forward and **turns in place**; executes the policy's velocity commands.
7. Safety floor cuts motors on stall/tip.

## 10. Open items (settle later, not blocking Phase 1)

- 2-wheel+caster vs 4-wheel tank, wheel type/diameter, top speed → by Roy's floors (Phase 2).
- Exact battery cell count + motor voltage + 5 V regulator → quick power budget (Phase 2);
  couples to the dock charge path (#3: 1S TP4056 vs 2S BMS).
- "Playfulness"/"politeness" reward weights → tuned in sim then on Roy.
- Perception, behavior arbiter, memory, dream, companion/voice, dock → sub-projects #2–#6
  (see the [vision & roadmap doc](2026-06-22-roybot-vision-and-roadmap.md)).

## 11. References

- GrowBot V0 — https://github.com/britcruise9/GrowBot
- Tan et al. 2018, "Sim-to-Real: Learning Agile Locomotion for Quadruped Robots"
- legged_gym (ETH RSL / Rudin et al.) — reward-term structure reference
- stable-baselines3 (PPO) · MuJoCo MJX / MuJoCo Playground (Brax PPO)
- Stanford Pupper v3 — closest DIY Pi + sim + RL→real analog
- N20 gearmotor + encoder · TB6612FNG dual H-bridge · TSOP38238 IR receiver + pogo pins (dock #3)
