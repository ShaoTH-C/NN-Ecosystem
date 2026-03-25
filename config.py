# config.py - all the tuneable knobs for the simulation live here

import numpy as np

# window / rendering
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
WORLD_WIDTH = 1400
WORLD_HEIGHT = 900
FPS = 60
BACKGROUND_COLOR = (15, 15, 25)

# simulation basics
INITIAL_HERBIVORES = 40
INITIAL_CARNIVORES = 12
INITIAL_FOOD = 120
MAX_FOOD = 300
FOOD_SPAWN_RATE = 2          # new food per tick
FOOD_ENERGY = 30.0
FOOD_RADIUS = 4

# creature movement / combat
CREATURE_RADIUS = 8
HERBIVORE_COLOR = (60, 200, 100)
CARNIVORE_COLOR = (220, 60, 60)
CREATURE_MAX_SPEED = 3.5      # fallback
HERBIVORE_MAX_SPEED = 3.0     # slower but can turn sharply
CARNIVORE_MAX_SPEED = 4.2     # faster so they can actually catch things
HERBIVORE_TURN_RATE = 0.18    # quick dodges
CARNIVORE_TURN_RATE = 0.12    # wider turns, more momentum
CREATURE_TURN_RATE = 0.15     # fallback
INITIAL_ENERGY = 80.0
MAX_ENERGY = 200.0
MOVE_ENERGY_COST = 0.12
IDLE_ENERGY_COST = 0.04
ATTACK_ENERGY_COST = 1.5
ATTACK_RANGE = 18.0
ATTACK_DAMAGE = 35.0
EAT_RANGE = 14.0

# reproduction
REPRODUCE_ENERGY_THRESHOLD = 140.0
REPRODUCE_ENERGY_COST = 60.0
REPRODUCE_COOLDOWN = 60       # ticks before they can reproduce again
MIN_POPULATION = 8            # safety net so species don't go fully extinct
MAX_POPULATION_HERBIVORE = 120
MAX_POPULATION_CARNIVORE = 50

# neural network layout
# 8 rays * 3 channels (food, herb, carn) + energy + speed + bias = 27 inputs
# outputs: turn, speed, eat, attack
NUM_SENSOR_RAYS = 8
SENSOR_RANGE = 150.0
NN_INPUT_SIZE = NUM_SENSOR_RAYS * 3 + 3
NN_HIDDEN_LAYERS = [20, 16]
NN_OUTPUT_SIZE = 4
NN_ACTIVATION = "tanh"

# evolution / mutation
MUTATION_RATE = 0.25
MUTATION_STRENGTH = 0.4       # gaussian noise std dev
WEIGHT_INIT_STD = 1.0
CROSSOVER_RATE = 0.3
TOPOLOGY_MUTATION_RATE = 0.05 # chance to add/remove a hidden neuron

# analytics
TRACK_INTERVAL = 30
PLOT_SAVE_PATH = "analytics_output"

# visualization colors
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
