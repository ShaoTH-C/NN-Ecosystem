# tracker.py - records simulation stats over time and generates plots at the end

import os
import numpy as np
import json
from typing import List, Dict, Optional
from datetime import datetime

from core.world import World
import config as cfg


class Tracker:
    """
    Takes periodic snapshots of the simulation state and stores them.
    Can dump everything to JSON and generate matplotlib plots for analysis.
    """

    def __init__(self, world: World):
        self.world = world
        self.history: List[Dict] = []
        self.last_record_tick = 0
        os.makedirs(cfg.PLOT_SAVE_PATH, exist_ok=True)

    def update(self):
        if self.world.tick - self.last_record_tick >= cfg.TRACK_INTERVAL:
            self.record()
            self.last_record_tick = self.world.tick

    def record(self):
        """Snapshot the current state."""
        stats = self.world.get_stats()

        herbs = self.world.herbivores
        carns = self.world.carnivores

        stats["avg_fitness_herb"] = (
            float(np.mean([c.genome.fitness for c in herbs])) if herbs else 0
        )
        stats["avg_fitness_carn"] = (
            float(np.mean([c.genome.fitness for c in carns])) if carns else 0
        )
        stats["max_fitness_herb"] = (
            float(max(c.genome.fitness for c in herbs)) if herbs else 0
        )
        stats["max_fitness_carn"] = (
            float(max(c.genome.fitness for c in carns)) if carns else 0
        )
        stats["avg_speed_herb"] = (
            float(np.mean([c.speed for c in herbs])) if herbs else 0
        )
        stats["avg_speed_carn"] = (
            float(np.mean([c.speed for c in carns])) if carns else 0
        )
        stats["total_creatures"] = len(herbs) + len(carns)

        if herbs:
            stats["median_gen_herb"] = int(
                np.median([c.genome.generation for c in herbs])
            )
        else:
            stats["median_gen_herb"] = 0

        if carns:
            stats["median_gen_carn"] = int(
                np.median([c.genome.generation for c in carns])
            )
        else:
            stats["median_gen_carn"] = 0

        self.history.append(stats)

    def save_history(self, filename: Optional[str] = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_history_{timestamp}.json"

        filepath = os.path.join(cfg.PLOT_SAVE_PATH, filename)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {filepath}")

    def generate_plots(self):
        """Make a nice multi-panel analysis figure with matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not installed, skipping plots.")
            return

        if len(self.history) < 2:
            print("Not enough data for plots.")
            return

        ticks = [h["tick"] for h in self.history]

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "NeuroEvolution Ecosystem - Simulation Analysis",
            fontsize=18,
            fontweight="bold",
            color="#E0E0E0",
        )
        fig.patch.set_facecolor("#1A1A2E")

        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
        style_kwargs = {"facecolor": "#1A1A2E"}

        # population over time
        ax1 = fig.add_subplot(gs[0, 0:2], **style_kwargs)
        ax1.plot(ticks, [h["herbivores"] for h in self.history],
                 color="#3CC864", linewidth=2, label="Herbivores")
        ax1.plot(ticks, [h["carnivores"] for h in self.history],
                 color="#DC3C3C", linewidth=2, label="Carnivores")
        ax1.plot(ticks, [h["food"] for h in self.history],
                 color="#50C832", linewidth=1, alpha=0.5, label="Food")
        ax1.set_title("Population Dynamics", color="#E0E0E0", fontsize=14)
        ax1.set_xlabel("Tick", color="#A0A0A0")
        ax1.set_ylabel("Count", color="#A0A0A0")
        ax1.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax1.tick_params(colors="#808090")
        ax1.set_facecolor("#12122A")
        ax1.grid(True, alpha=0.15)

        # predator/prey ratio
        ax2 = fig.add_subplot(gs[0, 2], **style_kwargs)
        ratios = []
        for h in self.history:
            if h["herbivores"] > 0:
                ratios.append(h["carnivores"] / h["herbivores"])
            else:
                ratios.append(0)
        ax2.plot(ticks, ratios, color="#FFB347", linewidth=2)
        ax2.axhline(y=0.3, color="#808090", linestyle="--", alpha=0.5, label="Ideal ratio")
        ax2.set_title("Predator-Prey Ratio", color="#E0E0E0", fontsize=14)
        ax2.set_xlabel("Tick", color="#A0A0A0")
        ax2.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax2.tick_params(colors="#808090")
        ax2.set_facecolor("#12122A")
        ax2.grid(True, alpha=0.15)

        # fitness curves
        ax3 = fig.add_subplot(gs[1, 0:2], **style_kwargs)
        ax3.plot(ticks, [h["avg_fitness_herb"] for h in self.history],
                 color="#3CC864", linewidth=2, label="Avg Herb Fitness")
        ax3.plot(ticks, [h["max_fitness_herb"] for h in self.history],
                 color="#3CC864", linewidth=1, alpha=0.4, linestyle="--",
                 label="Max Herb Fitness")
        ax3.plot(ticks, [h["avg_fitness_carn"] for h in self.history],
                 color="#DC3C3C", linewidth=2, label="Avg Carn Fitness")
        ax3.plot(ticks, [h["max_fitness_carn"] for h in self.history],
                 color="#DC3C3C", linewidth=1, alpha=0.4, linestyle="--",
                 label="Max Carn Fitness")
        ax3.set_title("Fitness Evolution", color="#E0E0E0", fontsize=14)
        ax3.set_xlabel("Tick", color="#A0A0A0")
        ax3.set_ylabel("Fitness", color="#A0A0A0")
        ax3.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0",
                   fontsize=9)
        ax3.tick_params(colors="#808090")
        ax3.set_facecolor("#12122A")
        ax3.grid(True, alpha=0.15)

        # average energy
        ax4 = fig.add_subplot(gs[1, 2], **style_kwargs)
        ax4.plot(ticks, [h["avg_energy_herb"] for h in self.history],
                 color="#3CC864", linewidth=2, label="Herbivores")
        ax4.plot(ticks, [h["avg_energy_carn"] for h in self.history],
                 color="#DC3C3C", linewidth=2, label="Carnivores")
        ax4.set_title("Average Energy", color="#E0E0E0", fontsize=14)
        ax4.set_xlabel("Tick", color="#A0A0A0")
        ax4.set_ylabel("Energy", color="#A0A0A0")
        ax4.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax4.tick_params(colors="#808090")
        ax4.set_facecolor("#12122A")
        ax4.grid(True, alpha=0.15)

        # generation progress
        ax5 = fig.add_subplot(gs[2, 0], **style_kwargs)
        ax5.plot(ticks, [h["max_gen_herb"] for h in self.history],
                 color="#3CC864", linewidth=2, label="Herbivore")
        ax5.plot(ticks, [h["max_gen_carn"] for h in self.history],
                 color="#DC3C3C", linewidth=2, label="Carnivore")
        ax5.set_title("Max Generation", color="#E0E0E0", fontsize=14)
        ax5.set_xlabel("Tick", color="#A0A0A0")
        ax5.set_ylabel("Generation", color="#A0A0A0")
        ax5.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax5.tick_params(colors="#808090")
        ax5.set_facecolor("#12122A")
        ax5.grid(True, alpha=0.15)

        # births/deaths/kills cumulative
        ax6 = fig.add_subplot(gs[2, 1], **style_kwargs)
        ax6.plot(ticks, [h["total_births"] for h in self.history],
                 color="#47B5FF", linewidth=2, label="Births")
        ax6.plot(ticks, [h["total_deaths"] for h in self.history],
                 color="#FF6B6B", linewidth=2, label="Deaths")
        ax6.plot(ticks, [h["total_kills"] for h in self.history],
                 color="#FFB347", linewidth=2, label="Kills")
        ax6.set_title("Cumulative Events", color="#E0E0E0", fontsize=14)
        ax6.set_xlabel("Tick", color="#A0A0A0")
        ax6.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax6.tick_params(colors="#808090")
        ax6.set_facecolor("#12122A")
        ax6.grid(True, alpha=0.15)

        # average speed per species
        ax7 = fig.add_subplot(gs[2, 2], **style_kwargs)
        ax7.plot(ticks, [h["avg_speed_herb"] for h in self.history],
                 color="#3CC864", linewidth=2, label="Herbivores")
        ax7.plot(ticks, [h["avg_speed_carn"] for h in self.history],
                 color="#DC3C3C", linewidth=2, label="Carnivores")
        ax7.set_title("Average Speed", color="#E0E0E0", fontsize=14)
        ax7.set_xlabel("Tick", color="#A0A0A0")
        ax7.set_ylabel("Speed", color="#A0A0A0")
        ax7.legend(facecolor="#2A2A3E", edgecolor="#404060", labelcolor="#E0E0E0")
        ax7.tick_params(colors="#808090")
        ax7.set_facecolor("#12122A")
        ax7.grid(True, alpha=0.15)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(cfg.PLOT_SAVE_PATH, f"analysis_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
        plt.close()
        print(f"Analysis plots saved to {filepath}")

    def print_summary(self):
        if not self.history:
            print("No data recorded.")
            return

        latest = self.history[-1]
        print("\n" + "=" * 60)
        print("  SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Duration:          {latest['tick']:,} ticks")
        print(f"  Herbivores:        {latest['herbivores']}")
        print(f"  Carnivores:        {latest['carnivores']}")
        print(f"  Total Births:      {latest['total_births']:,}")
        print(f"  Total Deaths:      {latest['total_deaths']:,}")
        print(f"  Total Kills:       {latest['total_kills']:,}")
        print(f"  Max Gen (Herb):    {latest['max_gen_herb']}")
        print(f"  Max Gen (Carn):    {latest['max_gen_carn']}")
        print(f"  Best Fitness (H):  {latest.get('max_fitness_herb', 0):.1f}")
        print(f"  Best Fitness (C):  {latest.get('max_fitness_carn', 0):.1f}")
        print("=" * 60 + "\n")
