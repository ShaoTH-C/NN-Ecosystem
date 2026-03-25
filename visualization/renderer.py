# renderer.py - pygame visualization for the ecosystem
# draws creatures, food, sensors, HUD stats, and population graphs in real-time

import pygame
import numpy as np
from typing import List, Tuple, Optional

from core.creature import Creature, Species
from core.food import Food
from core.world import World
import config as cfg


class Renderer:
    def __init__(self, world: World):
        pygame.init()
        pygame.display.set_caption("NeuroEvolution Ecosystem Simulator")

        self.win_w = cfg.WINDOW_WIDTH
        self.win_h = cfg.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.world = world
        self._rebuild_fonts()

        # rendering toggles
        self.show_sensors = False
        self.show_hud = True
        self.show_energy_bars = True
        self.selected_creature: Optional[Creature] = None
        self.paused = False
        self.sim_speed = 1
        self.turbo = False  # uncapped fps + high speed

        # graph history
        self.herb_history: List[int] = []
        self.carn_history: List[int] = []
        self.food_history: List[int] = []
        self.graph_max_points = 300

        self.food_surface = self._create_food_surface()

    def _rebuild_fonts(self):
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_medium = pygame.font.SysFont("consolas", 16)
        self.font_large = pygame.font.SysFont("consolas", 22)
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coords to screen coords (handles window resizing)."""
        sx = int(wx * self.win_w / cfg.WORLD_WIDTH)
        sy = int(wy * self.win_h / cfg.WORLD_HEIGHT)
        return sx, sy

    def _scale(self, val: float) -> int:
        """Scale a world-space size to screen-space."""
        return max(1, int(val * min(self.win_w / cfg.WORLD_WIDTH,
                                     self.win_h / cfg.WORLD_HEIGHT)))

    def _create_food_surface(self) -> pygame.Surface:
        """Pre-render a little glowing food sprite."""
        size = cfg.FOOD_RADIUS * 4
        surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        center = (size, size)
        for r in range(size, 0, -1):
            alpha = int(80 * (r / size))
            color = (*cfg.COLOR_FOOD, alpha)
            pygame.draw.circle(surf, color, center, r)
        pygame.draw.circle(surf, cfg.COLOR_FOOD, center, cfg.FOOD_RADIUS)
        return surf

    # --- main render loop ---

    def render(self) -> bool:
        """Draw one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self._handle_event(event)

        self.screen.fill(cfg.BACKGROUND_COLOR)
        self._draw_grid()
        self._draw_food()
        self._draw_creatures()

        if self.selected_creature and self.selected_creature.alive:
            self._draw_selected_info()

        if self.show_hud:
            self._draw_hud()
            self._draw_population_graph()

        self._draw_controls_help()

        pygame.display.flip()
        if self.turbo:
            self.clock.tick(0)   # uncapped
        else:
            self.clock.tick(cfg.FPS)
        return True

    # --- input handling ---

    def _handle_event(self, event: pygame.event.Event):
        if event.type == pygame.VIDEORESIZE:
            self.win_w = max(event.w, 400)
            self.win_h = max(event.h, 300)
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), pygame.RESIZABLE
            )
            self.food_surface = self._create_food_surface()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_s:
                self.show_sensors = not self.show_sensors
            elif event.key == pygame.K_h:
                self.show_hud = not self.show_hud
            elif event.key == pygame.K_e:
                self.show_energy_bars = not self.show_energy_bars
            elif event.key == pygame.K_UP:
                self.sim_speed = min(self.sim_speed + 1, 50)
            elif event.key == pygame.K_DOWN:
                self.sim_speed = max(self.sim_speed - 1, 1)
            elif event.key == pygame.K_ESCAPE:
                self.selected_creature = None
            elif event.key == pygame.K_f:
                self.turbo = not self.turbo
                if self.turbo:
                    self.sim_speed = 30
            # quick speed presets
            elif event.key == pygame.K_1:
                self.sim_speed = 1; self.turbo = False
            elif event.key == pygame.K_2:
                self.sim_speed = 2; self.turbo = False
            elif event.key == pygame.K_3:
                self.sim_speed = 5; self.turbo = False
            elif event.key == pygame.K_4:
                self.sim_speed = 10
            elif event.key == pygame.K_5:
                self.sim_speed = 20
            elif event.key == pygame.K_6:
                self.sim_speed = 50; self.turbo = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._select_creature(event.pos)

    def _select_creature(self, pos: Tuple[int, int]):
        mx, my = pos
        best_creature = None
        best_dist = 30

        for creature in self.world.creatures:
            sx, sy = self._world_to_screen(creature.x, creature.y)
            dx = sx - mx
            dy = sy - my
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_creature = creature

        self.selected_creature = best_creature

    # --- drawing ---

    def _draw_grid(self):
        grid_color = (25, 25, 40)
        spacing = 80
        for x in range(0, self.win_w, spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.win_h))
        for y in range(0, self.win_h, spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.win_w, y))

    def _draw_food(self):
        for food in self.world.food_items:
            if not food.alive:
                continue
            sx, sy = self._world_to_screen(food.x, food.y)
            r = cfg.FOOD_RADIUS * 4
            self.screen.blit(
                self.food_surface,
                (sx - r, sy - r),
            )

    def _draw_creatures(self):
        for creature in self.world.creatures:
            if not creature.alive:
                continue

            cx, cy = self._world_to_screen(creature.x, creature.y)
            r = self._scale(cfg.CREATURE_RADIUS)

            if self.show_sensors or creature is self.selected_creature:
                self._draw_sensors(creature)

            # body
            body_color = creature.color
            if creature is self.selected_creature:
                pygame.draw.circle(
                    self.screen, (255, 255, 100), (cx, cy), r + 4, 2
                )

            brightness = max(0.0, min(1.0, 0.4 + 0.6 * (creature.energy / cfg.MAX_ENERGY)))
            drawn_color = tuple(
                int(c * brightness) for c in body_color
            )
            pygame.draw.circle(self.screen, drawn_color, (cx, cy), r)

            inner_color = tuple(min(c + 40, 255) for c in drawn_color)
            pygame.draw.circle(self.screen, inner_color, (cx, cy), max(1, r // 2))

            # direction line
            end_x = cx + int(np.cos(creature.angle) * (r + 5))
            end_y = cy + int(np.sin(creature.angle) * (r + 5))
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (end_x, end_y), 2)

            # energy bar
            if self.show_energy_bars:
                bar_w = r * 2
                bar_h = 3
                bar_x = cx - r
                bar_y = cy - r - 6
                energy_frac = max(0.0, min(1.0, creature.energy / cfg.MAX_ENERGY))
                pygame.draw.rect(
                    self.screen, (40, 40, 40),
                    (bar_x, bar_y, bar_w, bar_h),
                )
                fill_w = int(bar_w * energy_frac)
                if fill_w > 0:
                    fill_color = (
                        int(255 * (1 - energy_frac)),
                        int(255 * energy_frac),
                        0,
                    )
                    pygame.draw.rect(
                        self.screen, fill_color,
                        (bar_x, bar_y, fill_w, bar_h),
                    )

            # generation label
            if creature.genome.generation > 0:
                gen_text = self.font_small.render(
                    str(creature.genome.generation), True, (180, 180, 200)
                )
                self.screen.blit(gen_text, (cx + r + 2, cy - 6))

    def _draw_sensors(self, creature: Creature):
        cx, cy = self._world_to_screen(creature.x, creature.y)

        for i, (endpoint, hit) in enumerate(
            zip(creature.sensor_endpoints, creature.sensor_hits)
        ):
            end = self._world_to_screen(endpoint[0], endpoint[1])
            ray_color = (50, 50, 70)
            pygame.draw.line(self.screen, ray_color, (cx, cy), end, 1)

            if hit is not None:
                hit_pos = self._world_to_screen(hit[0], hit[1])
                idx = i * 3
                readings = creature.sensor_readings
                if readings[idx + 0] < 1.0:
                    pygame.draw.circle(self.screen, cfg.COLOR_FOOD, hit_pos, 3)
                if readings[idx + 1] < 1.0:
                    pygame.draw.circle(
                        self.screen, cfg.HERBIVORE_COLOR, hit_pos, 3
                    )
                if readings[idx + 2] < 1.0:
                    pygame.draw.circle(
                        self.screen, cfg.CARNIVORE_COLOR, hit_pos, 3
                    )

    # --- HUD ---

    def _draw_hud(self):
        stats = self.world.get_stats()

        self.herb_history.append(stats["herbivores"])
        self.carn_history.append(stats["carnivores"])
        self.food_history.append(stats["food"])
        if len(self.herb_history) > self.graph_max_points:
            self.herb_history.pop(0)
            self.carn_history.pop(0)
            self.food_history.pop(0)

        hud_w = 280
        hud_h = 340
        hud_x = self.win_w - hud_w - 10
        hud_y = 10

        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill((20, 20, 35, 200))
        self.screen.blit(hud_surf, (hud_x, hud_y))
        pygame.draw.rect(
            self.screen, (60, 60, 80),
            (hud_x, hud_y, hud_w, hud_h), 1,
        )

        title = self.font_large.render("ECOSYSTEM", True, (200, 200, 220))
        self.screen.blit(title, (hud_x + 10, hud_y + 8))

        tick_text = self.font_small.render(
            f"Tick: {stats['tick']:,}", True, (150, 150, 170)
        )
        self.screen.blit(tick_text, (hud_x + hud_w - 110, hud_y + 12))

        y = hud_y + 38
        line_h = 20

        lines = [
            ("--- Population ---", (100, 100, 120)),
            (f"  Herbivores:  {stats['herbivores']}", cfg.HERBIVORE_COLOR),
            (f"  Carnivores:  {stats['carnivores']}", cfg.CARNIVORE_COLOR),
            (f"  Food:        {stats['food']}", cfg.COLOR_FOOD),
            ("", None),
            ("--- Energy ---", (100, 100, 120)),
            (f"  Avg Herb:  {stats['avg_energy_herb']:.1f}", cfg.HERBIVORE_COLOR),
            (f"  Avg Carn:  {stats['avg_energy_carn']:.1f}", cfg.CARNIVORE_COLOR),
            ("", None),
            ("--- Evolution ---", (100, 100, 120)),
            (f"  Gen (H/C):   {stats['max_gen_herb']}/{stats['max_gen_carn']}", (180, 180, 200)),
            (f"  Births:      {stats['total_births']:,}", (180, 180, 200)),
            (f"  Deaths:      {stats['total_deaths']:,}", (180, 180, 200)),
            (f"  Kills:       {stats['total_kills']:,}", (220, 80, 80)),
        ]

        for text, color in lines:
            if text == "":
                y += 4
                continue
            if color is None:
                color = cfg.COLOR_HUD_TEXT
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (hud_x + 10, y))
            y += line_h

    def _draw_population_graph(self):
        graph_w = 280
        graph_h = 100
        graph_x = self.win_w - graph_w - 10
        graph_y = 360

        graph_surf = pygame.Surface((graph_w, graph_h), pygame.SRCALPHA)
        graph_surf.fill((20, 20, 35, 200))
        self.screen.blit(graph_surf, (graph_x, graph_y))
        pygame.draw.rect(
            self.screen, (60, 60, 80),
            (graph_x, graph_y, graph_w, graph_h), 1,
        )

        title = self.font_small.render("Population History", True, (150, 150, 170))
        self.screen.blit(title, (graph_x + 10, graph_y + 4))

        if len(self.herb_history) < 2:
            return

        all_vals = self.herb_history + self.carn_history + self.food_history
        max_val = max(max(all_vals), 1)
        padding_top = 20
        padding_bottom = 5
        usable_h = graph_h - padding_top - padding_bottom

        def to_points(history, color):
            n = len(history)
            step = max(1, (graph_w - 20) / max(n - 1, 1))
            points = []
            for i, val in enumerate(history):
                px = graph_x + 10 + int(i * step)
                py = graph_y + padding_top + usable_h - int(
                    (val / max_val) * usable_h
                )
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)

        to_points(self.food_history, cfg.COLOR_GRAPH_FOOD)
        to_points(self.herb_history, cfg.COLOR_GRAPH_HERB)
        to_points(self.carn_history, cfg.COLOR_GRAPH_CARN)

    def _draw_selected_info(self):
        c = self.selected_creature
        if c is None or not c.alive:
            return

        panel_w = 250
        panel_h = 200
        panel_x = 10
        panel_y = self.win_h - panel_h - 10

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((20, 20, 35, 220))
        self.screen.blit(panel_surf, (panel_x, panel_y))
        pygame.draw.rect(
            self.screen, (100, 100, 130),
            (panel_x, panel_y, panel_w, panel_h), 1,
        )

        species_name = c.species.name.title()
        title = self.font_medium.render(
            f"Selected: {species_name} #{c.id}", True, c.color
        )
        self.screen.blit(title, (panel_x + 10, panel_y + 8))

        y = panel_y + 32
        lines = [
            f"Energy:    {c.energy:.1f} / {cfg.MAX_ENERGY}",
            f"Age:       {c.age} ticks",
            f"Speed:     {c.speed:.2f}",
            f"Gen:       {c.genome.generation}",
            f"Children:  {c.children_count}",
            f"Food:      {c.food_eaten}",
            f"Kills:     {c.kills}",
            f"Fitness:   {c.genome.fitness:.1f}",
            f"Brain:     {c.genome.layer_sizes}",
        ]
        for line in lines:
            text = self.font_small.render(line, True, (180, 180, 200))
            self.screen.blit(text, (panel_x + 10, y))
            y += 17

    def _draw_controls_help(self):
        speed_label = f"Speed: {self.sim_speed}x"
        if self.turbo:
            speed_label += " TURBO"
        controls = [
            "[SPACE] Pause",
            "[S] Sensors",
            "[H] HUD",
            "[E] Energy bars",
            f"[1-6] {speed_label}",
            "[F] Turbo toggle",
            "[Click] Select",
        ]
        x = 10
        y = 10
        for text in controls:
            rendered = self.font_small.render(text, True, (80, 80, 100))
            self.screen.blit(rendered, (x, y))
            y += 16

        if self.paused:
            pause_text = self.font_title.render("PAUSED", True, (255, 200, 50))
            self.screen.blit(
                pause_text,
                (self.win_w // 2 - 60, self.win_h // 2 - 14),
            )

    def close(self):
        pygame.quit()
