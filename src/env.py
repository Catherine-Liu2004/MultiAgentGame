import random
from typing import List, Tuple

from config import (
    ACTIONS,
    ATTACK_DAMAGE_ADJ,
    ATTACK_DAMAGE_SAME,
    COOP_BONUS_DIST,
    ENABLE_MOVING_OBSTACLES,
    MOVING_OBS_STEP_INTERVAL,
    MULTI_ATTACK_BONUS,
    PLAYER_MAX_HP,
    PLAYER_REWARD_COOP_TRAP,
    PLAYER_REWARD_DAMAGE_SCALE,
    PLAYER_REWARD_HIT_PENALTY,
    PLAYER_REWARD_STEP_SURVIVE,
    PLAYER_REWARD_SUCCESS_ESCAPE,
    REWARD_COLLISION,
    REWARD_COOP,
    REWARD_HIT_PLAYER,
    REWARD_KILL_PLAYER,
    REWARD_STEP,
)

Pos = Tuple[int, int]


class MultiNpcEnv:
    def __init__(self, npc_positions: List[Pos], player_position: Pos, walls: set, map_h: int, map_w: int):
        self.init_npc_positions = list(npc_positions)
        self.init_player_position = player_position

        self.map_h = map_h
        self.map_w = map_w

        # 静态墙体来自地图文件
        self.static_walls = set(walls)

        # 动态障碍
        self.dynamic_obstacles = []
        self.step_count = 0

        # 运行时状态
        self.npc_positions = list(npc_positions)
        self.player_position = player_position
        self.player_hp = PLAYER_MAX_HP
        self.last_damage = 0
        self.last_hits = 0

        self._init_dynamic_obstacles()
        self._refresh_walls()

    def _init_dynamic_obstacles(self):
        # 针对 map_02(13x9) 设计的默认巡逻点；小地图会自动过滤
        candidates = [
            {"pos": (2, 2), "dir": (0, 1)},
            {"pos": (2, 10), "dir": (0, -1)},
            {"pos": (6, 2), "dir": (0, 1)},
            {"pos": (6, 10), "dir": (0, -1)},
        ]
        self.dynamic_obstacles = []
        for obs in candidates:
            p = obs["pos"]
            if self._in_bounds(p) and (p not in self.static_walls):
                self.dynamic_obstacles.append(obs)

    def _refresh_walls(self):
        # 当前可阻挡格 = 静态墙 + 动态障碍
        self.walls = set(self.static_walls)
        for obs in self.dynamic_obstacles:
            self.walls.add(obs["pos"])

    def reset(self):
        self.npc_positions = list(self.init_npc_positions)
        self.player_position = self.init_player_position
        self.player_hp = PLAYER_MAX_HP
        self.last_damage = 0
        self.last_hits = 0
        self.step_count = 0

        self._init_dynamic_obstacles()
        self._refresh_walls()
        return self.get_states()

    def get_states(self):
        hp_bucket = self.player_hp // 10
        return [npc + self.player_position + (hp_bucket,) for npc in self.npc_positions]

    def get_player_state(self):
        """
        给玩家网络的全局状态：
        [player_x, player_y, hp_ratio,
         npc1_x, npc1_y, dist1,
         npc2_x, npc2_y, dist2, ...
         up_walkable, down_walkable, left_walkable, right_walkable,
         ul_walkable, ur_walkable, dl_walkable, dr_walkable,
         step_ratio]

        所有位置/距离都做归一化，避免不同地图尺度下输入幅值过大，
        也减少 DQN 训练时对绝对坐标的依赖。
        """
        px, py = self.player_position
        pos_x_scale = max(1.0, float(self.map_h - 1))
        pos_y_scale = max(1.0, float(self.map_w - 1))
        dist_scale = max(1.0, float(self.map_h + self.map_w - 2))

        state = [px / pos_x_scale, py / pos_y_scale, self.player_hp / PLAYER_MAX_HP]

        for npc in self.npc_positions:
            nx, ny = npc
            d = self.manhattan(self.player_position, npc)
            state.extend([nx / pos_x_scale, ny / pos_y_scale, d / dist_scale])

        neighbors = []
        for a in ACTIONS:
            np_ = self.move(self.player_position, a)
            neighbors.append(0.0 if np_ == self.player_position else 1.0)
        state.extend(neighbors)

        max_steps_assumed = 80.0
        state.append(self.step_count / max_steps_assumed)

        return state

    def _in_bounds(self, p: Pos):
        return 0 <= p[0] < self.map_h and 0 <= p[1] < self.map_w

    def _walkable(self, p: Pos):
        return self._in_bounds(p) and (p not in self.walls)

    @staticmethod
    def manhattan(a: Pos, b: Pos):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def move(self, pos: Pos, action: str):
        x, y = pos
        dx, dy = 0, 0

        if action == "up":
            dx, dy = -1, 0
        elif action == "down":
            dx, dy = 1, 0
        elif action == "left":
            dx, dy = 0, -1
        elif action == "right":
            dx, dy = 0, 1
        elif action == "up_left":
            dx, dy = -1, -1
        elif action == "up_right":
            dx, dy = -1, 1
        elif action == "down_left":
            dx, dy = 1, -1
        elif action == "down_right":
            dx, dy = 1, 1

        nx, ny = x + dx, y + dy
        np_ = (nx, ny)

        if not self._walkable(np_):
            return pos

        # 防止斜向穿墙
        if dx != 0 and dy != 0:
            side1 = (x + dx, y)
            side2 = (x, y + dy)
            if (side1 in self.walls) and (side2 in self.walls):
                return pos

        return np_

    def _move_dynamic_obstacles(self):
        occupied_npc = set(self.npc_positions)
        player = self.player_position

        for obs in self.dynamic_obstacles:
            x, y = obs["pos"]
            dx, dy = obs["dir"]
            nxt = (x + dx, y + dy)

            blocked = (
                (not self._in_bounds(nxt))
                or (nxt in self.static_walls)
                or (nxt in occupied_npc)
                or (nxt == player)
            )

            if blocked:
                dx, dy = -dx, -dy
                obs["dir"] = (dx, dy)
                nxt = (x + dx, y + dy)

                blocked_again = (
                    (not self._in_bounds(nxt))
                    or (nxt in self.static_walls)
                    or (nxt in occupied_npc)
                    or (nxt == player)
                )
                if blocked_again:
                    nxt = (x, y)

            obs["pos"] = nxt

        self._refresh_walls()

    def player_evasive_move(self):
        """
        规则玩家：
        选择一个能让自己离“最近 NPC”尽量远的位置。
        """
        candidates = []
        for a in ACTIONS:
            np_ = self.move(self.player_position, a)
            nearest = min(self.manhattan(np_, npc) for npc in self.npc_positions)
            candidates.append((nearest, np_))

        if not candidates:
            return

        best = max(score for score, _ in candidates)
        choices = [p for score, p in candidates if score == best]
        self.player_position = random.choice(choices)

    def player_random_move(self):
        """
        随机玩家：
        每步随机选一个动作。
        """
        action = random.choice(ACTIONS)
        self.player_position = self.move(self.player_position, action)

    def player_controlled_move(self, player_action_index: int):
        """
        ADV 玩家：
        由 DQN 输出动作索引。
        """
        action = ACTIONS[player_action_index]
        self.player_position = self.move(self.player_position, action)

    def step(self, action_indices: List[int], player_action_index=None, player_mode="rule", survive=False):
        """
        player_mode:
            - "random": 随机玩家
            - "rule": 规则逃跑玩家
            - "adv": DQN 对抗玩家

        返回:
            next_states, npc_rewards, player_reward, done, info
        """
        # 动态障碍先移动
        self.step_count += 1
        if ENABLE_MOVING_OBSTACLES and MOVING_OBS_STEP_INTERVAL > 0:
            if self.step_count % MOVING_OBS_STEP_INTERVAL == 0:
                self._move_dynamic_obstacles()

        # 1) NPC移动
        new_positions = []
        collision_rewards = [0] * len(self.npc_positions)

        for i, npc in enumerate(self.npc_positions):
            a = ACTIONS[action_indices[i]]
            np_ = self.move(npc, a)
            if np_ == npc and npc != self.player_position:
                collision_rewards[i] = REWARD_COLLISION
            new_positions.append(np_)

        self.npc_positions = new_positions

        # 2) 攻击判定（先按当前玩家位置结算）
        damages = []
        hit_indices = []
        for i, npc in enumerate(self.npc_positions):
            d = self.manhattan(npc, self.player_position)
            if d == 0:
                damages.append((i, ATTACK_DAMAGE_SAME))
                hit_indices.append(i)
            elif d == 1:
                damages.append((i, ATTACK_DAMAGE_ADJ))
                hit_indices.append(i)

        hit_count = len(damages)
        total_damage = 0
        for _, base_dmg in damages:
            dmg = base_dmg + (MULTI_ATTACK_BONUS if hit_count >= 2 else 0)
            total_damage += dmg

        self.player_hp -= total_damage
        self.last_damage = total_damage
        self.last_hits = hit_count

        # 3) NPC奖励
        distances = [self.manhattan(npc, self.player_position) for npc in self.npc_positions]
        coop_bonus = REWARD_COOP if all(d <= COOP_BONUS_DIST for d in distances) else 0

        done = self.player_hp <= 0
        npc_rewards = []
        for i in range(len(self.npc_positions)):
            r = REWARD_STEP + collision_rewards[i] + coop_bonus
            if i in hit_indices:
                r += REWARD_HIT_PLAYER
            if done and i in hit_indices:
                r += REWARD_KILL_PLAYER
            npc_rewards.append(r)

        # 4) 玩家移动 + 玩家奖励（给 adv 玩家）
        # 先执行玩家动作，再根据动作前后变化计算 reward，使训练信号更直接。
        prev_player_pos = self.player_position
        if not done:
            if player_mode == "adv" and player_action_index is not None:
                self.player_controlled_move(player_action_index)
            elif player_mode == "random":
                self.player_random_move()
            else:
                self.player_evasive_move()

        after_distances = [self.manhattan(self.player_position, npc) for npc in self.npc_positions]
        min_dist_before = min(distances) if len(distances) > 0 else 0
        min_dist_after = min(after_distances) if len(after_distances) > 0 else 0

        player_reward = PLAYER_REWARD_STEP_SURVIVE
        player_reward -= PLAYER_REWARD_DAMAGE_SCALE * total_damage
        player_reward -= PLAYER_REWARD_HIT_PENALTY * hit_count

        # 更直接地鼓励“本步真的逃远了”
        player_reward += 1.0 * (min_dist_after - min_dist_before)
        player_reward += 0.1 * min_dist_after

        if all(d <= COOP_BONUS_DIST for d in after_distances):
            player_reward -= PLAYER_REWARD_COOP_TRAP

        if min_dist_after == 0:
            player_reward -= 6.0
        elif min_dist_after == 1:
            player_reward -= 2.0

        if self.player_position == prev_player_pos and not done:
            player_reward -= 1.0

        if survive and not done:
            player_reward += PLAYER_REWARD_SUCCESS_ESCAPE

        next_states = self.get_states()
        info = {
            "player_hp": self.player_hp,
            "last_damage": self.last_damage,
            "last_hits": self.last_hits,
            "done_by_kill": done,
            "player_state": self.get_player_state(),
            "min_dist_before": min_dist_before,
            "min_dist_after": min_dist_after,
        }

        return next_states, npc_rewards, player_reward, done, info

        return next_states, npc_rewards, player_reward, done, info