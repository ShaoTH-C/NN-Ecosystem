# main.py - entry point for Ecosystem: a god game built on a neuro-evolutionary world
#
# Two visual modes share the same simulation:
#   • GAME mode (default)  — stylized "god view" with a left-hand toolbar of
#     divine actions: bring food, manifest creatures, bless the land, summon
#     rain, strike meteors, cast plagues. Designed for presentation / play.
#   • DEBUG mode           — the original technical renderer with sensors,
#     raw stats, and graphs. Useful for inspecting the underlying NN behavior.
#
# usage:
#   python main.py                   # game mode
#   python main.py --mode debug      # technical view
#   python main.py --headless        # no window, just crunch numbers
#   python main.py --ticks 5000      # run N ticks then save plots
#
# in-window:
#   TAB         swap between game mode and debug mode
#   SPACE       pause/resume
#   1-7         pick a divine tool (game mode)
#   click       use the active tool / select a creature
#   right-click inspect creature (game mode)
#   F           turbo speed
#   1-6         speed presets (debug mode)
#   S           toggle sensor rays
#   H           toggle HUD
#   Q           quit and save analytics

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.world import World
from analytics.tracker import Tracker
import config as cfg


def _make_renderer(mode: str, world: World):
    if mode == "debug":
        from visualization.renderer import Renderer
        return Renderer(world)
    from visualization.game_renderer import GameRenderer
    return GameRenderer(world)


def run_visual(max_ticks: int = 0, mode: str = "game", show_intro: bool = True):
    import pygame
    from core.sound import get_sounds

    # Title screen first (only in game mode — debug view skips intro).
    if show_intro and mode == "game":
        from visualization.intro import run_intro
        result = run_intro()
        if result == "quit":
            pygame.quit()
            return

    world = World()
    renderer = _make_renderer(mode, world)
    tracker = Tracker(world)

    # swap intro music → in-game music (no-op if files missing).
    sounds = get_sounds()
    if mode == "game":
        sounds.play_music(cfg.SOUND_MUSIC_MAIN, loop=True, fade_ms=900)
    else:
        sounds.stop_music()

    print("=" * 60)
    print(f"  ECOSYSTEM: {('GOD GAME' if mode == 'game' else 'DEBUG VIEW')}")
    print("=" * 60)
    print(f"  Initial herbivores:  {cfg.INITIAL_HERBIVORES}")
    print(f"  Initial carnivores:  {cfg.INITIAL_CARNIVORES}")
    print(f"  Neural network:      {cfg.get_nn_architecture()}")
    print(f"  Total NN params:     {world.creatures[0].brain.total_params()}")
    print("=" * 60)
    if mode == "game":
        print("  Click a tool on the left, then click the world to use it.")
        print("  TAB = swap to debug view   SPACE = pause   Q = quit")
    else:
        print("  SPACE=pause  S=sensors  F=turbo  TAB=game view  Q=quit")
    print("=" * 60)

    # Fixed-timestep sim pacing: we maintain a wall-clock-stable simulation
    # rate (sim_speed × BASE_SIM_TPS) regardless of the render frame rate.
    # At each render frame, accumulate elapsed time and run however many sim
    # ticks should have happened by now (capped to avoid runaway catch-up).
    # Result: the world looks exactly as fast at 250 creatures as at 50.
    running = True
    current_mode = mode
    sim_accumulator = 0.0
    last_time = time.time()
    try:
        while running:
            now = time.time()
            dt = min(now - last_time, 0.25)  # clamp huge gaps (e.g., after pause)
            last_time = now

            if not renderer.paused:
                target_tps = cfg.BASE_SIM_TPS * renderer.sim_speed
                sim_accumulator += dt * target_tps
                ticks_to_run = min(int(sim_accumulator), cfg.MAX_SIM_TICKS_PER_FRAME)
                sim_accumulator -= ticks_to_run

                for i in range(ticks_to_run):
                    # only build sensor-viz data on the final tick before the
                    # next render — saves work during catch-up batches
                    world.store_sensor_viz = (i == ticks_to_run - 1)
                    world.step()
                    tracker.update()
                    if max_ticks > 0 and world.tick >= max_ticks:
                        running = False
                        break
            else:
                sim_accumulator = 0.0  # don't pile up while paused

            running = renderer.render() and running

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            # TAB: swap renderer mode without losing the simulation
            if keys[pygame.K_TAB]:
                new_mode = "debug" if current_mode == "game" else "game"
                renderer.close()
                current_mode = new_mode
                renderer = _make_renderer(current_mode, world)
                # match music to the new mode
                if current_mode == "game":
                    sounds.play_music(cfg.SOUND_MUSIC_MAIN, loop=True, fade_ms=600)
                else:
                    sounds.stop_music()
                pygame.time.wait(180)
                last_time = time.time()  # reset clock so the swap doesn't burst sim

    except KeyboardInterrupt:
        print("\nInterrupted.")

    sounds.stop_music()
    renderer.close()
    tracker.record()
    tracker.print_summary()
    tracker.save_history()
    tracker.generate_plots()


def run_headless(max_ticks: int = 100000):
    world = World()
    tracker = Tracker(world)

    print("=" * 60)
    print("  HEADLESS MODE")
    print(f"  Running for {max_ticks:,} ticks...")
    print("=" * 60)

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
        description="Ecosystem: a god game built on a neuro-evolutionary world"
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run without a window (just crunch numbers)")
    parser.add_argument("--mode", choices=["game", "debug"], default=cfg.DEFAULT_MODE,
                        help="Visual mode (default: game)")
    parser.add_argument("--ticks", type=int, default=0,
                        help="Max ticks (0 = unlimited in visual mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-intro", action="store_true",
                        help="Skip the title screen (jump straight into the world)")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        print(f"  Seed: {args.seed}")

    if args.headless:
        ticks = args.ticks if args.ticks > 0 else 10000
        run_headless(ticks)
    else:
        run_visual(args.ticks, mode=args.mode, show_intro=not args.no_intro)


if __name__ == "__main__":
    main()
