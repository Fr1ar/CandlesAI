import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parser import parse_level, generate_default_level
from utils import log_action, render_pretty_colored

MAX_BLOCKS = 15
FEATURES_PER_BLOCK = 6
OBS_SIZE = MAX_BLOCKS * FEATURES_PER_BLOCK


class PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, text_level=None, max_steps=50):
        super().__init__()
        self.width = 6
        self.height = 6
        self.max_steps = max_steps

        self.text_level = text_level
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.step_num = 0
        self.total_steps = 0
        self.last_action = None
        self.prev_actions = []
        self.logging_enabled = True

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # Дискретные действия: MAX_BLOCKS*2 (каждый блок можно двигать в двух направлениях)
        self.action_space = spaces.Discrete(MAX_BLOCKS * 2)

    def set_logging_enabled(self, enabled: bool):
        self.logging_enabled = enabled

    # ----------------- MASK FOR MaskablePPO -----------------
    def action_mask(self):
        """
        Returns 1D mask of length action_space.n (MAX_BLOCKS * 2).
        Each action index corresponds to: action = block_index * 2 + direction
        direction: 0 = backward (left/up), 1 = forward (right/down)
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        block_items = list(self.blocks.items())
        num_blocks = len(block_items)

        for idx in range(num_blocks):
            block_id, _ = block_items[idx]
            for direction in (0, 1):
                if self._can_move(block_id, direction):
                    mask[idx * 2 + direction] = 1
        return mask

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.last_action = None
        self.step_num = 0
        self.prev_actions = []

        if self.text_level is None:
            self.blocks, self.block_texts, self.key_id = generate_default_level()
        else:
            self.blocks, self.block_texts, self.key_id = parse_level(text_level=self.text_level)

        if self.logging_enabled:
            print("\n=== Начальное состояние уровня ===")
            render_pretty_colored(self)

        return self._get_obs(), {}

    # ----------------- observation -----------------
    def _get_obs(self):
        arr = np.zeros((MAX_BLOCKS, FEATURES_PER_BLOCK), dtype=np.float32)
        for i, (block_id, block) in enumerate(self.blocks.items()):
            if i >= MAX_BLOCKS:
                break
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            type_H = 1.0 if block["type"] == "H" else 0.0
            is_key = 1.0 if block_id == self.key_id else 0.0

            arr[i, 0] = x / (self.width - 1)
            arr[i, 1] = y / (self.height - 1)
            arr[i, 2] = w / self.width
            arr[i, 3] = h / self.height
            arr[i, 4] = type_H
            arr[i, 5] = is_key
        return arr.flatten()

    def _index_to_block_id(self, chosen_index):
        block_items = list(self.blocks.items())
        if 0 <= chosen_index < len(block_items):
            return block_items[chosen_index][0]
        return None

    def _can_move(self, block_id, direction):
        block = self.blocks[block_id]
        dx = (1 if direction == 1 else -1) if block["type"] == "H" else 0
        dy = (1 if direction == 1 else -1) if block["type"] == "V" else 0

        # bounds check
        if block["x"] + dx < 0 or block["x"] + block["w"] + dx > self.width:
            return False
        if block["y"] + dy < 0 or block["y"] + block["h"] + dy > self.height:
            return False

        # collision check
        new_x = block["x"] + dx
        new_y = block["y"] + dy
        for oid, other in self.blocks.items():
            if oid == block_id:
                continue
            if (new_x < other["x"] + other["w"] and
                new_x + block["w"] > other["x"] and
                new_y < other["y"] + other["h"] and
                new_y + block["h"] > other["y"]):
                return False
        return True

    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"] == "H":
            block["x"] += 1 if direction == 1 else -1
        else:
            block["y"] += 1 if direction == 1 else -1

    def _compute_reward(self, block_id, direction, moved, violated, is_reverse, is_solved, terminated, invalid_action=False):
        reward = 0
        if moved:
            reward += 0.05
        if invalid_action:
            reward -= 0.3
        if self.last_action is not None and self.last_action[0] == block_id and self.last_action[1] == direction:
            reward += 0.10
        if violated:
            reward -= 0.6
        if is_reverse:
            reward -= 0.4
        if is_solved:
            reward += 10.0
        if not is_solved and terminated:
            reward -= 5.0
        return reward

    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"] + key["w"] - 1 == (self.width - 1)

    def step(self, action):
        # convert discrete action to block_index + direction
        if isinstance(action, (tuple, list, np.ndarray)):
            action = int(np.asarray(action).flatten()[0])
        action = int(action)

        chosen_index = action // 2
        direction = action % 2

        real_block_id = self._index_to_block_id(chosen_index)
        invalid_action = real_block_id is None

        moved = False
        violated = False
        is_reverse = False

        if not invalid_action:
            is_reverse = (
                self.last_action is not None and
                self.last_action[0] == real_block_id and
                self.last_action[1] != direction
            )
            if self._can_move(real_block_id, direction):
                self._move_block(real_block_id, direction)
                moved = True
            else:
                violated = True

        is_solved = self._is_solved()
        terminated = is_solved or self.step_num >= self.max_steps - 1
        truncated = False

        reward = self._compute_reward(
            real_block_id if real_block_id is not None else -1,
            direction, moved, violated, is_reverse, is_solved, terminated, invalid_action
        )

        self.prev_actions.append(f"{real_block_id}:{direction}")

        if self.logging_enabled:
            log_action(action, self, moved, self.step_num, self.total_steps, reward)

        self.step_num += 1
        self.total_steps += 1
        self.last_action = (real_block_id, direction) if not invalid_action else None

        obs = self._get_obs()
        info = {"is_success": is_solved}
        return obs, reward, terminated, truncated, info
