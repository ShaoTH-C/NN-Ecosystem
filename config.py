# config.py - all the tuneable knobs for the simulation live here

import numpy as np

# ---- GPU acceleration ----
# set True to use CuPy (requires NVIDIA GPU + cupy-cudaXXx installed)
# falls back to numpy automatically if cupy is unavailable
USE_GPU = True

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
INITIAL_FOOD = 150
MAX_FOOD = 350
FOOD_SPAWN_RATE = 3          # new food per tick
FOOD_ENERGY = 45.0
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
MAX_ENERGY = 400.0
BASE_METABOLISM = 0.08        # energy/tick just for being alive
MOVE_ENERGY_EXTRA = 0.03     # additional energy/tick scaled by speed ratio
CARNIVORE_METABOLISM_MULT = 1.5  # carnivores burn more (die faster without prey)

# ---- combat ----
ATTACK_RANGE = 25.0
ATTACK_DAMAGE = 60.0          # base damage (modified by speed)
ATTACK_ENERGY_COST = 0.3
INSTINCT_ATTACK_RANGE = 18.0  # auto-attack when this close (reflex)
EAT_RANGE = 15.0
INSTINCT_EAT_RANGE = 8.0      # auto-eat when this close
KILL_BASE_ENERGY = 80.0       # minimum energy from any kill
KILL_ENERGY_FRACTION = 0.4    # + this fraction of prey's current energy

# ---- maturity (children) ----
MATURATION_TICKS = 200         # ticks to reach full maturity
CHILD_SPEED_SCALE = 0.5       # newborn speed multiplier (grows to 1.0)
CHILD_INITIAL_ENERGY_FRAC = 0.5  # child starts with this fraction of breeding cost
CHILD_ENERGY_VALUE_MULT = 0.4    # energy value when a child is killed (discourages hunting babies)

# ---- reproduction / breeding ----
# no cooldown - requires finding a mate, being in age range, having energy
BREEDING_AGE_MIN_FRAC = 0.15  # earliest breeding age as fraction of lifespan
BREEDING_AGE_MAX_FRAC = 0.70  # latest breeding age
BREEDING_ENERGY_THRESHOLD = 130.0
BREEDING_ENERGY_COST = 60.0
MATE_SEARCH_RANGE = 200.0
MATE_AGE_TOLERANCE = 250      # mate's age must be within this many ticks

# knowledge transfer (parent -> child)
CHILD_MUTATION_RATE = 0.08    # much lower than normal mutation
CHILD_MUTATION_STRENGTH = 0.12

# population safety
MIN_POPULATION = 8
MAX_POPULATION_HERBIVORE = 300
MAX_POPULATION_CARNIVORE = 25

# ---- neural network ----
NUM_SENSOR_RAYS = 8
SENSOR_RANGE = 150.0

# Input layout (40 total):
#   [0-23]   8 rays x 3 channels (food_dist, herb_dist, carn_dist per ray)
#   [24]     energy / MAX_ENERGY
#   [25]     speed / max_speed
#   [26]     age / lifespan
#   [27]     maturity (0-1, 0=newborn, 1=adult)
#   [28-30]  nearest food:          sin(rel_angle), cos(rel_angle), dist/range
#   [31-33]  nearest same-species:  sin, cos, dist  (for mate-finding / herding)
#   [34-36]  nearest threat/prey:   sin, cos, dist  (herbs->carn, carns->herb)
#   [37-38]  parent direction:      sin, cos  (fades as creature matures)
#   [39]     bias (always 1.0)
NN_INPUT_SIZE = 40
NN_HIDDEN_LAYERS = [48, 32]
NN_OUTPUT_SIZE = 5   # turn, speed, eat, attack, breed
NN_ACTIVATION = "tanh"

# ---- evolution / mutation ----
MUTATION_RATE = 0.20
MUTATION_STRENGTH = 0.35
WEIGHT_INIT_STD = 1.0
CROSSOVER_RATE = 0.5
TOPOLOGY_MUTATION_RATE = 0.0  # disabled for batched NN processing

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
