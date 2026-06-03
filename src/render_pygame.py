import os
import pygame

from config import HUD_WIDTH, TILE_SIZE


def safe_load(path, size):
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
    return None


class PygameRenderer:
    def __init__(self, map_lines, title="Coop Hunt"):
        pygame.init()
        pygame.font.init()

        self.map_lines = map_lines
        self.h = len(map_lines)
        self.w = len(map_lines[0])

        self.screen_w = self.w * TILE_SIZE + HUD_WIDTH
        self.screen_h = self.h * TILE_SIZE
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(title)

        # 稳定起见，使用pygame内置字体
        self.font = pygame.font.Font(None, 34)
        self.small = pygame.font.Font(None, 24)

        # 可选贴图（没有也能跑）
        self.tile_floor = safe_load("assets/floor.png", (TILE_SIZE, TILE_SIZE))
        self.tile_wall = safe_load("assets/wall.png", (TILE_SIZE, TILE_SIZE))
        self.player_img = safe_load("assets/player.png", (TILE_SIZE - 8, TILE_SIZE - 8))
        self.npc_img = safe_load("assets/npc.png", (TILE_SIZE - 8, TILE_SIZE - 8))
        self.dynamic_obs_img = safe_load("assets/moving_obstacle.png", (TILE_SIZE - 20, TILE_SIZE - 20))

        # 受击特效状态
        self.hit_flash_timer = 0
        self.floating_texts = []  # [{"x":..., "y":..., "text":"-20", "life":25}, ...]

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def draw_map(self):
        for i in range(self.h):
            for j in range(self.w):
                x, y = j * TILE_SIZE, i * TILE_SIZE
                c = self.map_lines[i][j]

                if c == "#":
                    if self.tile_wall:
                        self.screen.blit(self.tile_wall, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (95, 100, 110), (x, y, TILE_SIZE, TILE_SIZE))
                        pygame.draw.rect(self.screen, (70, 74, 82), (x + 4, y + 4, TILE_SIZE - 8, TILE_SIZE - 8), 2)
                else:
                    if self.tile_floor:
                        self.screen.blit(self.tile_floor, (x, y))
                    else:
                        base = (220, 225, 232) if (i + j) % 2 == 0 else (210, 216, 224)
                        pygame.draw.rect(self.screen, base, (x, y, TILE_SIZE, TILE_SIZE))

                pygame.draw.rect(self.screen, (180, 185, 192), (x, y, TILE_SIZE, TILE_SIZE), 1)

    def draw_entities(self, npc_positions, player_pos):
        # 玩家
        px, py = player_pos
        rx, ry = py * TILE_SIZE + 4, px * TILE_SIZE + 4
        pcx = py * TILE_SIZE + TILE_SIZE // 2
        pcy = px * TILE_SIZE + TILE_SIZE // 2

        if self.player_img:
            self.screen.blit(self.player_img, (rx, ry))
        else:
            pygame.draw.circle(self.screen, (66, 133, 244), (pcx, pcy), TILE_SIZE // 3)

        # 受击闪红圈
        if self.hit_flash_timer > 0:
            pygame.draw.circle(self.screen, (255, 50, 50), (pcx, pcy), TILE_SIZE // 2, 4)
            self.hit_flash_timer -= 1

        # NPC
        for idx, (nx, ny) in enumerate(npc_positions):
            ex, ey = ny * TILE_SIZE + 4, nx * TILE_SIZE + 4
            ecx = ny * TILE_SIZE + TILE_SIZE // 2
            ecy = nx * TILE_SIZE + TILE_SIZE // 2

            if self.npc_img:
                self.screen.blit(self.npc_img, (ex, ey))
            else:
                pygame.draw.circle(self.screen, (220, 70, 70), (ecx, ecy), TILE_SIZE // 3)

            tag = self.small.render(str(idx), True, (255, 255, 255))
            self.screen.blit(tag, (ecx - 6, ecy - 8))

    def draw_moving_obstacles(self, moving_obstacles):
        # 动态障碍：高亮显示，确保肉眼明显
        for (ox, oy) in moving_obstacles:
            rx, ry = oy * TILE_SIZE + 10, ox * TILE_SIZE + 10
            cx, cy = oy * TILE_SIZE + TILE_SIZE // 2, ox * TILE_SIZE + TILE_SIZE // 2
            r = TILE_SIZE // 3

            if self.dynamic_obs_img:
                self.screen.blit(self.dynamic_obs_img, (rx, ry))
            else:
                pygame.draw.circle(self.screen, (255, 170, 0), (cx, cy), r)
                pygame.draw.circle(self.screen, (255, 60, 0), (cx, cy), r, 3)
                pygame.draw.line(self.screen, (120, 20, 0), (cx - r // 2, cy), (cx + r // 2, cy), 3)
                pygame.draw.line(self.screen, (120, 20, 0), (cx, cy - r // 2), (cx, cy + r // 2), 3)

    def draw_hud(self, episode, step, algo, hp, last_damage, last_hits, success_rate):
        x0 = self.w * TILE_SIZE
        pygame.draw.rect(self.screen, (28, 32, 40), (x0, 0, HUD_WIDTH, self.screen_h))

        title = self.font.render("COOP HUNT HUD", True, (255, 255, 255))
        self.screen.blit(title, (x0 + 16, 16))

        lines = [
            f"Algo: {algo.upper()}",
            f"Episode: {episode}",
            f"Step: {step}",
            "",
            f"Player HP: {hp}",
            f"Last Damage: {last_damage}",
            f"Hit Count: {last_hits}",
            "",
            f"Success: {success_rate:.2%}",
        ]

        y = 60
        for t in lines:
            surf = self.small.render(t, True, (240, 240, 240))
            self.screen.blit(surf, (x0 + 16, y))
            y += 28

        # HP 条
        bar_x, bar_y, bar_w, bar_h = x0 + 16, 205, HUD_WIDTH - 32, 18
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))
        hp_ratio = max(0.0, min(1.0, hp / 100.0))
        color = (70, 200, 120) if hp > 50 else ((230, 180, 60) if hp > 20 else (230, 70, 70))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

    def draw_floating_texts(self):
        alive = []
        for t in self.floating_texts:
            surf = self.small.render(t["text"], True, (255, 80, 80))
            self.screen.blit(surf, (t["x"], t["y"]))
            t["y"] -= 1
            t["life"] -= 1
            if t["life"] > 0:
                alive.append(t)
        self.floating_texts = alive

    def draw(self, npc_positions, player_pos, moving_obstacles, episode, step, algo, hp, last_damage, last_hits, success_rate):
        # 触发受击特效
        if last_damage > 0:
            self.hit_flash_timer = 6
            px, py = player_pos
            x = py * TILE_SIZE + TILE_SIZE // 2 - 10
            y = px * TILE_SIZE + TILE_SIZE // 2 - 10
            self.floating_texts.append({"x": x, "y": y, "text": f"-{last_damage}", "life": 25})

        self.screen.fill((245, 247, 250))
        self.draw_map()
        self.draw_entities(npc_positions, player_pos)
        self.draw_moving_obstacles(moving_obstacles)
        self.draw_hud(episode, step, algo, hp, last_damage, last_hits, success_rate)
        self.draw_floating_texts()
        pygame.display.flip()

    @staticmethod
    def close():
        pygame.quit()