import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import log_action
import random


class PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, text_level=None, max_steps=50):
        super().__init__()
        self.width = 6
        self.height = 6
        self.step_num = 0
        self.max_steps = max_steps
        self.blocks = {}
        self.key_id = None
        self.last_key_x = 0
        self.state = None
        self.block_texts = {}
        self.text_level = text_level

        # annealing
        self.initial_exploration = 0
        self.exploration_prob = 0

        # For reverse-move penalty
        self.last_action = None

        # Will be overwritten after parse/reset
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6 * 6 * 3,),
            dtype=np.float32,
        )

    # ----------------- annealing exploration -----------------
    def _update_exploration_prob(self):
        t = self.step_num
        if t < 100_000:
            self.exploration_prob = 0.10
        elif t < 300_000:
            self.exploration_prob = 0.05
        else:
            self.exploration_prob = 0.005

    # ----------------- generate default level -----------------
    def generate_default_level(self):
        next_id = 0
        self.blocks[next_id] = {"x": 0, "y": 2, "w": 2, "h": 1, "type": "H"}
        self.block_texts[next_id] = "0"
        self.key_id = next_id
        next_id += 1

        self.blocks[next_id] = {"x": 2, "y": 0, "w": 3, "h": 1, "type": "H"}
        self.block_texts[next_id] = "a"
        next_id += 1

        self.blocks[next_id] = {"x": 3, "y": 1, "w": 1, "h": 3, "type": "V"}
        self.block_texts[next_id] = "b"
        next_id += 1

    # ----------------- parse text level -----------------
    def parse_level(self):
        next_id = 0
        lines = self.text_level.split(".")
        grid = [row.split() for row in lines]

        H, W = len(grid), len(grid[0])
        used = [[False] * W for _ in range(H)]

        for y in range(H):
            for x in range(W):
                if grid[y][x] == "-" or used[y][x]:
                    continue

                ch = grid[y][x]

                # horizontal?
                w = 1
                while x + w < W and grid[y][x + w] == ch:
                    w += 1

                if w > 1:
                    self.blocks[next_id] = {"x": x, "y": y, "w": w, "h": 1, "type": "H"}
                    self.block_texts[next_id] = ch
                    if ch == "0":
                        self.key_id = next_id

                    for dx in range(w):
                        used[y][x + dx] = True

                    next_id += 1
                    continue

                # vertical?
                h = 1
                while y + h < H and grid[y + h][x] == ch:
                    h += 1

                if h > 1:
                    self.blocks[next_id] = {"x": x, "y": y, "w": 1, "h": h, "type": "V"}
                    self.block_texts[next_id] = ch
                    if ch == "0":
                        self.key_id = next_id

                    for dy in range(h):
                        used[y + dy][x] = True

                    next_id += 1
                    continue

                # single block
                self.blocks[next_id] = {
                    "x": x, "y": y, "w": 1, "h": 1,
                    "type": "H" if ch == "0" else "V"
                }
                self.block_texts[next_id] = ch
                if ch == "0":
                    self.key_id = next_id

                used[y][x] = True
                next_id += 1

        if self.key_id is None:
            raise ValueError("No key '0' found")

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.last_action = None
        self.step_num = 0

        self._update_exploration_prob()

        if self.text_level is None:
            self.generate_default_level()
        else:
            self.parse_level()

        key = self.blocks[self.key_id]
        self.last_key_x = key["x"]

        # NEW: dynamic Discrete action space
        num_blocks = len(self.blocks)
        self.action_space = spaces.Discrete(num_blocks * 2)

        self.state = self._get_obs()
        return self.state, {}

    # ----------------- obs -----------------
    def _get_obs(self):
        obs = np.zeros((6, 6, 3), dtype=np.float32)
        for block_id, block in self.blocks.items():
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            channel = 0 if block_id == self.key_id else (1 if block["type"] == "H" else 2)
            for dy in range(h):
                for dx in range(w):
                    obs[y + dy, x + dx, channel] = 1.0
        return obs.flatten()

    # ----------------- can move -----------------
    def _can_move(self, block_id, direction):
        block = self.blocks[block_id]
        dx = (1 if direction == 1 else -1) if block["type"] == "H" else 0
        dy = (1 if direction == 1 else -1) if block["type"] == "V" else 0

        # border check
        if block["x"] + dx < 0 or block["x"] + block["w"] + dx > self.width:
            return False
        if block["y"] + dy < 0 or block["y"] + block["h"] + dy > self.height:
            return False

        # collision check
        for other_id, other in self.blocks.items():
            if other_id == block_id:
                continue
            if (
                block["x"] + dx < other["x"] + other["w"]
                and block["x"] + block["w"] + dx > other["x"]
                and block["y"] + dy < other["y"] + other["h"]
                and block["y"] + block["h"] + dy > other["y"]
            ):
                return False

        return True

    # ----------------- move -----------------
    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"] == "H":
            block["x"] += 1 if direction == 1 else -1
        else:
            block["y"] += 1 if direction == 1 else -1

    # ----------------- reward -----------------
    def _compute_reward(self, block_id, moved, violated, is_reverse, is_solved):
        # Штраф за каждый ход, чтобы агент быстрее искал решение
        reward = -0.1

        # Награда за движение ключа вправо
        key = self.blocks[self.key_id]
        if moved and block_id == self.key_id and key["x"] > self.last_key_x:
            self.last_key_x = key["x"]
            reward = key["x"] * 1.25

        # Штраф за движение за границы уровня
        if violated:
            reward = -0.5

        # Штраф за движение туда-обратно
        if is_reverse:
            reward = -0.5

        # Награда за решение
        if is_solved:
            reward = 10

        # Штраф за ненайденное решение
        if not is_solved and self.step_num >= self.max_steps:
            reward = -10

        return reward

    # ----------------- is solved -----------------
    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"] + key["w"] - 1 == 5

    # ----------------- step -----------------
    def step(self, action):
        self._update_exploration_prob()

        # may override by random exploration
        if random.random() < self.exploration_prob:
            action = random.randrange(self.action_space.n)

        # decode action
        block_id = action // 2
        direction = action % 2

        # check reverse
        is_reverse = (
            self.last_action is not None
            and self.last_action[0] == block_id
            and self.last_action[1] != direction
        )

        # can move?
        violated = False
        moved = False

        if self._can_move(block_id, direction):
            self._move_block(block_id, direction)
            moved = True
        else:
            violated = True

        is_solved = self._is_solved()
        terminated = is_solved or self.step_num >= self.max_steps
        truncated = False

        # compute reward
        reward = self._compute_reward(block_id, moved, violated, is_reverse, is_solved)

        log_action((block_id, direction), self, moved, self.step_num, reward)

        self.last_action = (block_id, direction)
        self.step_num += 1

        return self._get_obs(), reward, terminated, truncated, is_solved, {}
