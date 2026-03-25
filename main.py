# main.py - entry point for the neuroevolution ecosystem simulator
#
# creatures with neural network brains (numpy, no frameworks) evolve through
# natural selection in a 2D world. herbivores eat food, carnivores hunt
# herbivores, and evolution happens continuously through birth/death/mutation.
#
# controls:
#   SPACE       pause/resume
#   S           toggle sensor rays
#   H           toggle HUD
#   E           toggle energy bars
#   F           turbo mode (uncapped fps)
#   1-6         speed presets (1x, 2x, 5x, 10x, 20x, 50x)
#   UP/DOWN     fine-tune speed
#   Click       select a creature
#   ESC         deselect
#   Q           quit and save analysis
#
# usage:
#   python main.py              # run with visualization
#   python main.py --headless   # no window, just crunch numbers
#   python main.py --ticks 5000 # run for N ticks then generate plots

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from analytics.tracker import Tracker
import config as cfg


def run_visual(max_ticks: int = 0):
    from visualization.renderer import Renderer

    world = World()
    renderer = Renderer(world)
    tracker = Tracker(world)

    print("=" * 55)
    print("  NEUROEVOLUTION ECOSYSTEM SIMULATOR")
    print("=" * 55)
    print(f"  Initial herbivores:  {cfg.INITIAL_HERBIVORES}")
    print(f"  Initial carnivores:  {cfg.INITIAL_CARNIVORES}")
    print(f"  Neural network:      {cfg.get_nn_architecture()}")
    print(f"  Total NN params:     {world.creatures[0].brain.total_params()}")
    print("=" * 55)
    print("  SPACE=pause  S=sensors  F=turbo  1-6=speed  Q=quit")
    print("=" * 55)

    import pygame

    running = True
    frame = 0
    try:
        while running:
            # at high speeds, batch sim steps and only render occasionally
            # this keeps the UI responsive without redrawing every tick batch
            if renderer.sim_speed <= 5:
                steps_per_loop = renderer.sim_speed
                render_interval = 1
            else:
                # do 5 steps per loop iteration, render every Nth iteration
                steps_per_loop = 5
                render_interval = max(1, renderer.sim_speed // steps_per_loop)

            if frame % render_interval == 0:
                running = renderer.render()
            else:
                running = renderer.process_events()

            if not renderer.paused:
                # only compute sensor viz data on the last step before a render
                for i in range(steps_per_loop):
                    is_last_step = (i == steps_per_loop - 1)
                    next_frame_renders = ((frame + 1) % render_interval == 0)
                    world.store_sensor_viz = is_last_step and next_frame_renders
                    world.step()
                    tracker.update()

                    if max_ticks > 0 and world.tick >= max_ticks:
                        running = False
                        break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False

            frame += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")

    renderer.close()
    tracker.record()
    tracker.print_summary()
    tracker.save_history()
    tracker.generate_plots()


def run_headless(max_ticks: int = 100000):
    world = World()
    tracker = Tracker(world)

    print("=" * 55)
    print("  HEADLESS MODE")
    print(f"  Running for {max_ticks:,} ticks...")
    print("=" * 55)

    start_time = time.time()

    for tick in range(max_ticks):
        world.step()
        tracker.update()

        if tick % 1000 == 0 and tick > 0:
            elapsed = time.time() - start_time
            tps = tick / elapsed
            stats = world.get_stats()
            print(
                f"  Tick {tick:>6,} | "
                f"H:{stats['herbivores']:>3} C:{stats['carnivores']:>3} "
                f"F:{stats['food']:>3} | "
                f"Gen H:{stats['max_gen_herb']:>3} C:{stats['max_gen_carn']:>3} | "
                f"{tps:.0f} tps"
            )

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s ({max_ticks / elapsed:.0f} ticks/sec)")

    tracker.record()
    tracker.print_summary()
    tracker.save_history()
    tracker.generate_plots()


def main():
    parser = argparse.ArgumentParser(
        description="NeuroEvolution Ecosystem Simulator"
    )
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--ticks", type=int, default=0, help="Max ticks (0 = unlimited in visual mode)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        print(f"  Seed: {args.seed}")

    if args.headless:
        ticks = args.ticks if args.ticks > 0 else 10000
        run_headless(ticks)
    else:
        run_visual(args.ticks)


if __name__ == "__main__":
    main()
