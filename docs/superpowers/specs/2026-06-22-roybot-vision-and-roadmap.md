# Roybot — Vision & Roadmap

**Status:** Living doc · **Date:** 2026-06-22 · **Owner:** becqu

## What Roybot is

A small, fast, durable autonomous robot that **plays with the cat (Roy) on her terms** and grows
into a **curious companion** — it explores on its own, builds a memory of Roy and what she likes,
"dreams" each night to learn, and tells becqu how Roy is doing. A wheeled re-imagining of
[GrowBot V0](https://github.com/britcruise9/GrowBot), keeping its spirit (cheap, learning,
curious, "a foundation model with a body") but scoped with ponytail/karpathy discipline.

## Guiding principles

- **Local-first autonomy.** All fast, embodied behavior — play, juking, exploring, leaving Roy
  alone, self-righting, docking — runs on the Pi with **no LLM in the loop**. Instant, private,
  works offline.
- **Consent-based play.** Roybot never pesters. It engages when Roy is willing and **gives her
  space when she isn't.** Learned in the chase reward, enforced by the behavior arbiter.
- **The LLM is for language & reflection only** — daily reports, talking with becqu, narrating
  "dreams." A cat toy doesn't need an LLM; a companion's *words* do.
- **Learn from experience.** Every day's interactions are logged and fed back via the nightly
  "dream" — GrowBot's "each run feeds the next training pass."
- **Privacy intent.** Roy's data stays home wherever feasible; the long-term plan removes the
  cloud entirely.

## System architecture

| Layer | Where it runs | Role |
|---|---|---|
| **Embodied autonomy** | Pi Zero 2 W (onboard) | Drive, consent-based chase/play policy (tiny MLP @50 Hz), curiosity-driven exploration, behavior arbiter (play / leave-alone / explore / go-dock), reflexes, interaction logging |
| **Language & reflection** | **Cloud now** (OpenRouter → DeepSeek V4 Flash, model swappable) → **local on an NVIDIA Spark laptop (~1 yr)** | Daily Roy reports, two-way voice conversation, dream narration. Behind one swappable LLM-client interface so cloud→Spark is a config change |
| **Home base** | becqu's PC (RTX 3070) now → Spark later | Stores recordings; runs the nightly **dream**: consolidate logs, update Roy-preference model, optionally fine-tune the chase policy on real interactions |

Roybot connects to home base over WiFi when home; runs fully autonomously (play/explore) when
home base is off.

## "Dream" = nightly consolidation + learning

While docked/charging, the home base processes the day's logs to: (1) update what Roy
likes/dislikes and her rhythms, (2) optionally fine-tune the chase policy on real interactions,
(3) generate becqu's daily report. Narrated back as "dreams." (Literal world-model/Dreamer-style
imagined rollouts = explicit later stretch goal, not now.)

## Sub-projects & build order

Each gets its own brainstorm → spec → plan → build cycle. Build #1 Phase-1 first (pure software).

1. **Drive + Chase core** — drive base + closed-form driver + MuJoCo twin + moody sim cat +
   consent-based chase policy + onboard runtime. *(Spec approved; Phase-1 sim is next to build.)*
2. **Perception + behavior** — see/track Roy + read her engagement; environment perception;
   local behavior arbiter; **curiosity-driven exploration.** *Local, no LLM.*
3. **Memory + preference learning** — log interactions/outcomes; learn Roy's likes/dislikes +
   schedule; feed the arbiter and the dream. *Local.*
4. **Home base** — recording storage + nightly dream/retrain (RTX 3070 now → Spark later) + the
   swappable LLM-client.
5. **Companion layer** — **voice both ways** (wake-word + STT + TTS), proactive daily Roy reports,
   Q&A; **interesting-event detection → recording → home-base storage** (the bonus, native here).
6. **Self-charging dock** — battery-low → IR homing → passive funnel → pogo-pin charge; docking
   **triggers the dream.**

## Hardware at a glance (evolving; details in each sub-project spec)

Pi Zero 2 W · 2× N20 gearmotors (+encoders) · TB6612FNG H-bridge · MPU-6050 IMU · OV5647 camera ·
INMP441 mic · MAX98357A amp + speaker · WS2812 LED ring · small LiPo · pogo-pin dock. Reuses
GrowBot's driver layer except the servo bus.
