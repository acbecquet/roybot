"""All shared physical and behavioral constants. DRY: import from here."""

# --- control / sim timing ---
CONTROL_HZ = 50
SIM_TIMESTEP = 0.002            # must match models/roybot.xml <option timestep>
N_SUBSTEPS = round((1.0 / CONTROL_HZ) / SIM_TIMESTEP)  # = 10
EPISODE_SECONDS = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ

# --- drivetrain geometry (Pololu HPCB 6V 100:1 N20 + 60 mm wheel + encoders, 6V rail; verified spec) ---
# Encoders let the Pi close a wheel-speed PID, so this sim velocity actuator has a real counterpart.
# Confirm wheel OD + mounted track from the assembled bot, then refit + re-train if they differ.
WHEEL_RADIUS = 0.03             # m (60 mm N20 wheel)
WHEEL_BASE = 0.11               # m (left-right wheel track, N20-mounted on the 3216)
MAX_WHEEL_RAD_S = 25.05         # rad/s = 239 rpm loaded (330 rpm free-run * 0.725 derate) at 6V

# --- action scaling (policy outputs in [-1, 1]) ---
ACTION_VFWD_MAX = 0.6           # m/s (N20 100:1 @6V realizes this with margin)
ACTION_VYAW_MAX = 6.0           # rad/s

# --- moody cat ---
BAND_MIN = 0.15                 # m — closer than this = crowding/pestering
BAND_MAX = 0.50                 # m — the "interesting play band" outer edge
BAND_CENTER = (BAND_MIN + BAND_MAX) / 2
WILLING_THRESHOLD = 0.40        # engagement >= this => cat wants to play
ENGAGE_GAIN = 0.8               # /s engagement gained during good in-band play
PESTER_RATE = 1.2               # /s engagement lost while being crowded
ENGAGE_DECAY = 0.05             # /s natural waning
CAT_FLEE_SPEED = 0.7            # m/s
CAT_FLEE_SKITTISH_FACTOR = 0.3  # scales _skittishness contribution to flee speed
CAT_WANDER_SPEED = 0.1          # m/s
CAT_WANDER_TIMER_S = (0.5, 1.5) # seconds to hold a wander heading
CAT_DART_PROB = 0.04            # per control step, when willing
CAT_ARENA_HALF = 2.0            # m — cat position clamped to ±this
JUKE_RATE_CAP = 0.5             # m/s — cap on approach_rate magnitude for juke reward

# --- termination ---
TIP_UPRIGHT_MIN = 0.5           # body +z world-component below this => tipped over

# --- reward weights ---
REWARD_WEIGHTS = {
    "engage": 1.0,       # willing + in band
    "juke": 0.3,         # willing + dynamic distance (anti-static)
    "give_space": 1.0,   # disinterested + retreating
    "pester": 1.5,       # disinterested + approaching (penalty)
    "energy": 0.02,      # ||action||^2 penalty
    "action_rate": 0.05, # ||action - prev_action||^2 penalty
    "tip": 10.0,         # one-time penalty on tip-over
}

# --- Phase 1.5: frame stacking ---
N_STACK = 6                      # observations stacked for motion/anticipation context

# --- Phase 1.5: rich cat behavior modes ---
CAT_STALK_SPEED = 0.12          # m/s slow circling creep
CAT_POUNCE_SPEED = 0.9          # m/s burst toward the robot
CAT_BAT_SPEED = 0.45            # m/s quick swipe
CAT_DART_SPEED = 0.7            # m/s erratic burst
CAT_MODE_MIN_S = 0.4            # min seconds a play mode is held
CAT_MODE_MAX_S = 1.6            # max seconds a play mode is held
# personality is sampled per episode in Cat.reset() from these ranges:
CAT_PLAYFULNESS_RANGE = (0.6, 1.4)
CAT_SKITTISHNESS_RANGE = (0.5, 1.5)
CAT_ATTENTION_RANGE = (0.5, 1.5)

# --- Phase 1.5: anticipation reward ---
ANTICIPATE_HORIZON = 0.4        # s ahead to predict the cat's position
REWARD_WEIGHTS["anticipate"] = 0.6   # reward closing on the predicted (lead) position

# --- Phase 1.5: domain randomization ranges (widened) + difficulty ---
DR_MASS = (0.7, 1.3)            # body-mass scale
DR_FRICTION = (0.6, 1.4)        # geom-friction scale
DR_MOTOR_GAIN = (0.8, 1.2)      # commanded-speed scale
DR_LATENCY_STEPS = (0, 4)       # control-step latency (exclusive high -> 0..3)
DIFFICULTY_RANGE = (0.0, 1.0)   # per-episode difficulty; scales cat speed/evasiveness
CAT_SPEED_SCALE_AT_MAX = 1.3    # cat speed multiplier at difficulty=1.0 (1.0 at difficulty=0)
