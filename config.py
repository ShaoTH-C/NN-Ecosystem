# config.py - all the tuneable knobs for the simulation live here

import numpy as np

# ---- GPU acceleration ----
# When ON, the hot per-tick math (sensor casting, pairwise social distances,
# batched NN forward) runs on cupy. Falls back to numpy automatically if cupy
# isn't installed.
#
# Measured break-even on an RTX 4060 with the default world size:
#   N <  ~150 creatures → numpy is faster (kernel launch overhead dominates)
#   N >= ~200 creatures → GPU is 5–10× faster (sensor + social), grows with N
# So we always run with `USE_GPU = True` but gate per-call: if the alive
# population is below GPU_MIN_POP we transparently take the numpy path.
USE_GPU = True
GPU_MIN_POP = 150

# ---- window / rendering ----
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
WORLD_WIDTH = 1400
WORLD_HEIGHT = 900

# Render frame cap. 30 is intentional: it stays consistent across population
# sizes (we measured ~38 sim TPS at 240 creatures, so 30 has comfortable
# headroom and the game feels the same in late-game as it does at start).
FPS = 30

# Base wall-clock sim TPS at sim_speed=1. Sim ticks are paced to wall clock,
# not to render frames — so the world always runs at this speed regardless of
# how fast or slow the renderer happens to be. sim_speed multiplies it.
BASE_SIM_TPS = 30
# safety cap: maximum sim ticks per render frame. Kept low so a slow frame
# can't snowball — if rendering takes 100ms, we won't try to run 6 sim ticks
# to "catch up" (which would take another 100ms and starve the next frame).
# At normal sim_speed this never triggers; only kicks in after a hiccup.
MAX_SIM_TICKS_PER_FRAME = 6
BACKGROUND_COLOR = (15, 15, 25)

# ---- simulation basics ----
INITIAL_HERBIVORES = 35
INITIAL_CARNIVORES = 5
INITIAL_FOOD = 150
MAX_FOOD = 260               # plenty on the map so creatures don't starve
FOOD_SPAWN_RATE = 3          # faster regrowth — less attrition from hunger
FOOD_ENERGY = 45.0           # bigger meals
FOOD_RADIUS = 8              # fatter berries — more readable in game view

# ---- lifespan / aging ----
MIN_LIFESPAN = 900
MAX_LIFESPAN = 1000
AGING_START_FRACTION = 0.5   # aging effects begin at this fraction of lifespan
OLD_AGE_SPEED_MULT = 0.35   # speed multiplier at end of life (1.0 = no reduction)
OLD_AGE_METABOLISM_MULT = 2.5  # metabolism multiplier at end of life

# ---- creature movement / combat ----
CREATURE_RADIUS = 8
HERBIVORE_COLOR = (60, 200, 100)
CARNIVORE_COLOR = (220, 60, 60)
CREATURE_MAX_SPEED = 3.5      # fallback
HERBIVORE_MAX_SPEED = 3.0
CARNIVORE_MAX_SPEED = 4.0
HERBIVORE_TURN_RATE = 0.18    # quick dodges
CARNIVORE_TURN_RATE = 0.15    # slightly wider turns
CREATURE_TURN_RATE = 0.15     # fallback

# ---- energy system (real-world inspired) ----
# base metabolism: you starve even standing still, faster if you move.
# Numbers tuned generously so the world stays populated even at the lower
# population cap — design game wants visible life, not constant attrition.
INITIAL_ENERGY = 240.0
MAX_ENERGY = 360.0
BASE_METABOLISM = 0.13        # energy/tick just for being alive (forgiving baseline)
MOVE_ENERGY_EXTRA = 0.06     # additional energy/tick scaled by speed ratio
CARNIVORE_METABOLISM_MULT = 1.25  # carnivores burn more (die faster without prey)

# ---- satiation (fullness-driven behavior switching) ----
SATIATION_THRESHOLD = 0.85   # fraction of MAX_ENERGY above which creature is "full"
                             # full creatures stop eating and seek mates instead

# ---- combat ----
ATTACK_RANGE = 25.0
ATTACK_DAMAGE = 60.0          # base damage (modified by speed)
ATTACK_ENERGY_COST = 0.3
INSTINCT_ATTACK_RANGE = 18.0  # auto-attack when this close (reflex)
EAT_RANGE = 22.0               # larger range so clumsy creatures can still feed
INSTINCT_EAT_RANGE = 14.0     # auto-eat when this close (larger = more forgiving)
KILL_BASE_ENERGY = 110.0      # minimum energy from any kill (carnivores need good payoff)
KILL_ENERGY_FRACTION = 0.5    # + this fraction of prey's current energy

# ---- maturity (children) ----
MATURATION_TICKS = 200         # ticks to reach full maturity
CHILD_SPEED_SCALE = 0.5       # newborn speed multiplier (grows to 1.0)
CHILD_INITIAL_ENERGY_FRAC = 0.5  # child starts with this fraction of breeding cost
CHILD_ENERGY_VALUE_MULT = 0.4    # energy value when a child is killed (discourages hunting babies)

# ---- reproduction / breeding ----
# no cooldown - requires finding a mate, being in age range, having energy
BREEDING_AGE_MIN_FRAC = 0.15  # earliest breeding age as fraction of lifespan
BREEDING_AGE_MAX_FRAC = 0.75  # latest breeding age
BREEDING_ENERGY_THRESHOLD = 110.0  # lower threshold so well-fed creatures breed more
BREEDING_ENERGY_COST = 65.0        # cheaper breeding so populations stay healthy
MATE_SEARCH_RANGE = 250.0          # wider search to find mates
MATE_AGE_TOLERANCE = 400           # mate's age must be within this many ticks

# knowledge transfer (parent -> child)
CHILD_MUTATION_RATE = 0.08    # much lower than normal mutation
CHILD_MUTATION_STRENGTH = 0.12

# population safety — capped so the game stays at ≥10 fps comfortably and the
# toolbar actions (blessing, plague, etc.) feel responsive. Total ceiling is
# 130 creatures, well below the threshold where Python loops in eat/attack/
# reproduce start to bog down render. MIN_POPULATION raised so the auto-spawn
# floor keeps the world feeling alive even after a brutal disaster/plague.
MIN_POPULATION = 10
MAX_POPULATION_HERBIVORE = 110
MAX_POPULATION_CARNIVORE = 22

# ---- neural network ----
NUM_SENSOR_RAYS = 8
SENSOR_RANGE = 150.0

# Input layout (44 total):
#   [0-23]   8 rays x 3 channels (food_dist, herb_dist, carn_dist per ray)
#   [24]     energy / MAX_ENERGY
#   [25]     speed / max_speed
#   [26]     age / lifespan
#   [27]     maturity (0-1, 0=newborn, 1=adult)
#   [28-30]  nearest food:          sin(rel_angle), cos(rel_angle), dist/range
#   [31-33]  nearest same-species:  sin, cos, dist  (for mate-finding / herding)
#   [34-36]  nearest threat/prey:   sin, cos, dist  (herbs->carn, carns->herb)
#   [37-38]  parent direction:      sin, cos  (fades as creature matures)
#   [39]     hunger_urgency:        1.0=starving, 0.0=full (nonlinear, drives food-seeking)
#   [40]     is_satiated:           1.0=full (above threshold), 0.0=hungry (drives breeding)
#   [41]     breeding_readiness:    1.0=eligible to breed, 0.0=not ready
#   [42]     nearby_mates_signal:   normalized count of compatible mates in range
#   [43]     bias (always 1.0)
NN_INPUT_SIZE = 44
NN_HIDDEN_LAYERS = [52, 36]   # slightly larger brain for more inputs
NN_OUTPUT_SIZE = 5   # turn, speed, eat, attack, breed
NN_ACTIVATION = "tanh"

# ---- evolution / mutation ----
MUTATION_RATE = 0.20
MUTATION_STRENGTH = 0.35
WEIGHT_INIT_STD = 1.0
CROSSOVER_RATE = 0.5
TOPOLOGY_MUTATION_RATE = 0.0  # disabled for batched NN processing

# ---- social learning / teaching ----
# best performers in a group teach nearby creatures by blending NN weights
TEACHING_INTERVAL = 15         # teach every N ticks
TEACHING_RADIUS = 180.0        # range for knowledge transfer
TEACHING_BLEND_RATE = 0.12     # base blend rate toward teacher's weights
TEACHING_MAX_BLEND = 0.15      # max blend per teaching event
TEACHING_MIN_PERF_GAP = 0.15   # teacher must score this much higher

# ---- behavioral instincts (override bad NN decisions) ----
# these are innate reflexes that keep creatures alive while their NN learns
IDLE_BREAK_TICKS = 15           # if idle this long, force movement
CIRCLE_BREAK_THRESHOLD = 1.8    # turn_accumulator above this = break the circle
                                # (1.8 ≈ 60% of max sustained turn — clear circle)
CIRCLE_LOCKOUT_TICKS = 30       # forced straight-line ticks after a circle is detected
                                # (just nudging fails — NN immediately resumes circling)
MIN_WANDER_SPEED_FRAC = 0.15    # minimum speed as fraction of max (prevents standing still)

# Displacement-based circle detection: the turn_accumulator above only catches
# consistently-biased turning. Slow drift-circles (where the NN alternates turn
# signs but still traces a loop) keep the accumulator near zero and slip through.
# Solution: every N ticks, compare current position to a snapshot from N ticks
# ago. If the creature has walked a lot of path but barely moved in net distance,
# it's going in circles — trigger a hard break.
CIRCLE_SNAPSHOT_INTERVAL = 80          # ticks between position snapshots
CIRCLE_MIN_PATH_TO_CHECK = 50.0        # must have walked this much path to qualify
CIRCLE_MAX_NET_DISPLACEMENT = 45.0     # net displacement below this = circling
CIRCLE_HARD_LOCKOUT_TICKS = 70         # forced-straight after a hard break
CIRCLE_HARD_BREAK_MIN_AGE = 40         # ignore newborns while NN still settles

# ---- analytics ----
TRACK_INTERVAL = 30
PLOT_SAVE_PATH = "analytics_output"

# ---- visualization colors ----
COLOR_FOOD = (80, 200, 50)
COLOR_SENSOR_RAY = (255, 255, 255, 40)
COLOR_HUD_BG = (20, 20, 35, 200)
COLOR_HUD_TEXT = (220, 220, 230)
COLOR_GRAPH_HERB = (60, 200, 100)
COLOR_GRAPH_CARN = (220, 60, 60)
COLOR_GRAPH_FOOD = (80, 200, 50)

# ---- game mode (Ecosystem: God Game) ----
# default starting mode: "game" = stylized god-mode UI, "debug" = the technical view
DEFAULT_MODE = "game"

# folder where AI-generated sprites live (relative to project root)
ASSETS_DIR = "assets"

# ---- audio ----
# Master switch + volumes. Sound files live in <ASSETS_DIR>/sounds/.
# If a file is missing, a procedural synth fallback plays so the game still
# has audio feedback before AI-generated tracks are added.
SOUND_ENABLED = True
MUSIC_VOLUME = 0.45
SFX_VOLUME = 0.45
SOUNDS_SUBDIR = "sounds"

# filenames to look for inside assets/sounds/ (use .ogg or .wav)
SOUND_MUSIC_MAIN = "music_main.ogg"     # looping in-game music
SOUND_MUSIC_INTRO = "music_intro.ogg"   # title-screen track (optional)
SOUND_SFX_CLICK = "sfx_click.ogg"
SOUND_SFX_FOOD = "sfx_food.ogg"
SOUND_SFX_HERB = "sfx_spawn_herb.ogg"
SOUND_SFX_CARN = "sfx_spawn_carn.ogg"
SOUND_SFX_BLESSING = "sfx_blessing.ogg"
SOUND_SFX_RAIN = "sfx_rain.ogg"
SOUND_SFX_DISASTER = "sfx_disaster.ogg"
SOUND_SFX_PLAGUE = "sfx_plague.ogg"

# game-mode palette (warmer, nature-inspired)
GAME_BG_TOP = (118, 178, 122)        # grassy green (top)
GAME_BG_BOTTOM = (88, 142, 96)       # deeper green (bottom)
GAME_SIDEBAR_BG = (35, 30, 45, 235)  # parchment-purple, slightly translucent
GAME_SIDEBAR_ACCENT = (210, 175, 110)
GAME_SIDEBAR_TEXT = (240, 232, 215)
GAME_PANEL_BG = (32, 28, 40, 220)
GAME_PANEL_BORDER = (210, 175, 110)
GAME_TITLE_GOLD = (240, 200, 120)
GAME_GRID_COLOR = (98, 158, 106, 90)

# god-tool radii (used for cursor preview ring)
DROP_FOOD_SPREAD = 60.0
DISASTER_RADIUS = 130.0
BLESSING_RADIUS = 140.0
PLAGUE_RADIUS = 160.0


def get_nn_architecture():
    return [NN_INPUT_SIZE] + NN_HIDDEN_LAYERS + [NN_OUTPUT_SIZE]

SEED = None  # set to an int if you want reproducible runs
if SEED is not None:
    np.random.seed(SEED)
