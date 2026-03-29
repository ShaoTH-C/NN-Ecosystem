# config.py - all the tuneable knobs for the simulation live here

import numpy as np

# ---- GPU acceleration ----
# set True to use CuPy (requires NVIDIA GPU + cupy-cudaXXx installed)
# falls back to numpy automatically if cupy is unavailable
USE_GPU = False  # CuPy not available; NumPy is fast enough for current population sizes

# ---- window / rendering ----
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
WORLD_WIDTH = 1400
WORLD_HEIGHT = 900
FPS = 60
BACKGROUND_COLOR = (15, 15, 25)

# ---- simulation basics ----
INITIAL_HERBIVORES = 40
INITIAL_CARNIVORES = 6
INITIAL_FOOD = 120
MAX_FOOD = 200               # scarcer food = real competition
FOOD_SPAWN_RATE = 2          # new food per tick (slower regrowth)
FOOD_ENERGY = 40.0
FOOD_RADIUS = 4

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
# base metabolism: you starve even standing still, faster if you move
INITIAL_ENERGY = 200.0
MAX_ENERGY = 350.0
BASE_METABOLISM = 0.18        # energy/tick just for being alive (must eat to survive)
MOVE_ENERGY_EXTRA = 0.07     # additional energy/tick scaled by speed ratio
CARNIVORE_METABOLISM_MULT = 1.3  # carnivores burn more (die faster without prey)

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
BREEDING_ENERGY_THRESHOLD = 120.0  # lower threshold so well-fed creatures breed more
BREEDING_ENERGY_COST = 80.0        # but breeding is expensive (real investment)
MATE_SEARCH_RANGE = 250.0          # wider search to find mates
MATE_AGE_TOLERANCE = 400           # mate's age must be within this many ticks

# knowledge transfer (parent -> child)
CHILD_MUTATION_RATE = 0.08    # much lower than normal mutation
CHILD_MUTATION_STRENGTH = 0.12

# population safety (high caps - let food scarcity and predation control naturally)
MIN_POPULATION = 6
MAX_POPULATION_HERBIVORE = 500
MAX_POPULATION_CARNIVORE = 80

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
IDLE_BREAK_TICKS = 15          # if idle this long, force movement
CIRCLE_BREAK_THRESHOLD = 3.0   # turn_accumulator above this = break the circle
MIN_WANDER_SPEED_FRAC = 0.15   # minimum speed as fraction of max (prevents standing still)

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


def get_nn_architecture():
    return [NN_INPUT_SIZE] + NN_HIDDEN_LAYERS + [NN_OUTPUT_SIZE]

SEED = None  # set to an int if you want reproducible runs
if SEED is not None:
    np.random.seed(SEED)
