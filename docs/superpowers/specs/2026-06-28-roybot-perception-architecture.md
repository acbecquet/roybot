# Roybot — Perception Architecture (Sub-project #2) — Design Spec

**Status:** Approved (design) · **Date:** 2026-06-28 · **Owner:** becqu

How Roybot turns a camera into the abstract cat signal the chase policy consumes — and the
staged path from "offload to the PC" to "fully standalone." Companion to the
[drive-chase-core design](2026-06-22-roybot-drive-chase-core-design.md) and the
[vision & roadmap](2026-06-22-roybot-vision-and-roadmap.md).

---

## 1. The problem

The trained policy's observation is **abstract**, not pixels: per control step it needs the cat's
**relative position** (robot frame), **relative velocity**, an **engagement** scalar (0–1), and —
new, from the reality check — a **valid/visible** flag so the policy can handle "cat lost." On the
real robot all of that must be produced from the **OV5647 camera**. The reality check (workflow
`wiv1ane3g`) established two hard facts:

- A Raspberry Pi Zero 2 W **cannot** run a real cat detector in real time (~1–2 fps, RAM-bound).
- The policy's reward was shaped on sim ground-truth (metric position, true velocity, clean
  engagement); the deployed obs must carry the **same information content** or the behaviors don't
  transfer. So perception is co-designed with an obs/reward reframe, not bolted on after.

## 2. The contract (stable across all stages)

Perception is a **swappable provider** behind one interface — the same pattern as the LLM client.
The policy never knows which backend is running.

```
class PerceptionProvider:
    def get_cat_state(self) -> CatState: ...

CatState = {
    "bearing":      float,   # rad, cat direction in robot frame (left/right) — ALWAYS recoverable
    "range":        float,   # m, distance estimate (from blob-area or depth or detector box size)
    "approach_rate":float,   # m/s, +closing / -opening (frame-to-frame range delta)
    "engagement":   float,   # 0..1
    "visible":      bool,    # False => cat lost; policy switches to search/idle
    "confidence":   float,   # 0..1, for smoothing / lost-detection
}
```

**Obs reframe (do this before Stage A deploy):** retrain the policy on obs built from these fields
— **bearing, range, approach/retreat sign, engagement, visible** — i.e. quantities EVERY backend
below can produce. Drop any obs that needs metric self-motion (no encoders→no odometry into obs).
Add a trained **cat-lost path** (`visible=False` → defined search/idle, engagement decays). Widen
latency domain-randomization to **250–500 ms** and down-sample the cat signal to **5–10 Hz** in sim
so the policy matches the real pipeline. This supersedes `final_policy.npz`.

**Engagement is a heuristic, not a mood reading** (true affect from monocular video is research-
grade). v1 proxy: rises with sustained in-band interactive motion (cat approaching/batting), falls
when crowded or when the cat leaves / ignores. Tunable. Honest scope cut, documented as such.

## 3. Stages (B and C build on A — same interface, swap the backend)

### Stage A — Offload to PC (first prototype)
- **Backend:** bot streams camera frames over WiFi to becqu's PC (RTX 3070); PC runs a real cat
  detector + tracker, returns `CatState`. Bot runs only the thin client + the policy.
- **Hardware:** none beyond the Stage-A BOM (camera + Pi already there). **$0 extra.**
- **Pros:** sidesteps the Pi-Zero vision limit; can use a strong detector immediately.
- **Cons:** needs WiFi + home; adds LAN latency (tens of ms — inside the trained envelope);
  bot is tethered to the PC. On dropout → `visible=False` (the cat-lost path).
- **Success:** bot visibly tracks + plays with Roy with perception served from the PC.

### Stage B — Standalone onboard, Roy's white chest (cuts the PC cord)
- **Backend:** **HSV blob tracker on the Pi Zero** at 5–15 Hz locked onto **Roy's natural white
  chest patch** (she is all black with a clear white chest — a large, high-contrast blob, bigger
  and blobbier than a collar and always on her): segment the white region → centroid = bearing,
  **blob area = range proxy** (calibrate against the measured chest size), frame-to-frame Δ =
  approach_rate; hold + constant-velocity estimate up to the 50 Hz the policy wants. Heuristic
  engagement.
- **Hardware:** **same bot — NO new hardware at all** (uses her markings; no collar needed).
- **Pros:** fully standalone, no PC, no WiFi, no collar; light enough to run onboard.
- **Cons:** relies on her white chest being visible — drops on occlusion / bad light / when she
  faces fully away (→ `visible=False`, must be handled); tuned to Roy specifically.
- **Success:** plays with Roy with the PC off and WiFi irrelevant.

### Stage C — OPTIONAL: on-device AI detector (robustness upgrade)
- **When:** only if Stage B's white-chest blob proves too fragile (bad lighting, chest occluded,
  she faces away, or a second cat appears). Stage B already delivers no-collar standalone, so this
  is a robustness upgrade, **not a requirement**.
- **Backend:** a real on-device cat detector (e.g. quantized MobileNet-SSD / YOLO-nano) at useful
  fps — an AI image model that finds Roy with no marker, robust to lighting/occlusion; optionally
  a learned engagement estimate.
- **Hardware:** brain upgrade — **Pi 5 + an AI accelerator** (Coral USB Edge TPU ~$60, or the Pi
  AI HAT / Hailo-8L ~$70) + a larger battery. (Jetson Orin Nano is overkill — skip.) Everything
  else (drivetrain, sensors, audio, dock) carries over.
- **Cons:** +cost, power, weight; only worth it if B's tracking is insufficient.
- **Success:** detects + plays with Roy onboard regardless of whether her chest is visible.

## 4. Why this ordering

Each stage is a backend swap behind §2's interface, so the policy and the rest of the robot never
change. A gets a working, *capable* bot fastest using the PC (RTX 3070) for vision; B reaches true
standalone on the *same* hardware with **no new parts** by tracking Roy's white chest (the end goal
becqu wants); C is an **optional** robustness upgrade (an AI detector) only if the chest blob proves
too fragile. The obs/reward reframe in §2 is done once, up front, and holds for all three because
the contract fields are the lowest common denominator every backend can produce.

## 5. Open items
- Calibrate blob-area → range against the measured size of Roy's white chest patch (Stage B).
- Choose the Stage-A wire protocol (frames up / CatState down) + a watchdog → `visible=False` on
  stall. Keep raw frames off any cloud; PC offload is LAN-only.
- Engagement heuristic weights — tune in sim, then on Roy.
- Stage C detector + accelerator choice — defer until B is proven.
