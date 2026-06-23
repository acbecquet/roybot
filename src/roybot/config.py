"""All shared physical and behavioral constants. DRY: import from here."""

# --- control / sim timing ---
CONTROL_HZ = 50
SIM_TIMESTEP = 0.002            # must match models/roybot.xml <option timestep>
N_SUBSTEPS = round((1.0 / CONTROL_HZ) / SIM_TIMESTEP)  # = 10
EPISODE_SECONDS = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ

# --- drivetrain geometry ---
WHEEL_RADIUS = 0.025            # m
WHEEL_BASE = 0.09               # m (left-right wheel separation)
MAX_WHEEL_RAD_S = 30.0          # rad/s motor speed cap

# --- action scaling (policy outputs in [-1, 1]) ---
ACTION_VFWD_MAX = 0.6           # m/s
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
CAT_WANDER_SPEED = 0.1          # m/s
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
