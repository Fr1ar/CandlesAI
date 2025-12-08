import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parser import parse_level, generate_default_level
from utils import log_action, log_level, log_action_mask
from numba import njit

MAX_BLOCKS = 15
FEATURES_PER_BLOCK = 6
OBS_SIZE = MAX_BLOCKS * FEATURES_PER_BLOCK
GRID_SIZE = 6

# ----------------- NUMBA COLLISION CHECK -----------------
@njit
def check_collision_numba(x, y, w, h, dx, dy,
                          all_x, all_y, all_w, all_h,
                          block_idx, num_blocks):
    if x + dx < 0 or x + w + dx > GRID_SIZE:
        return False
    if y + dy < 0 or y + h + dy > GRID_SIZE:
        return False
    for i in range(num_blocks):
        if i == block_idx:
            continue
        ox, oy, ow, oh = all_x[i], all_y[i], all_w[i], all_h[i]
        if ox == -1:
            continue
        if x + dx < ox + ow and x + w + dx > ox and y + dy < oy + oh and y + h + dy > oy:
            return False
    return True

# ----------------- NUMBA CAN_MOVE -----------------
@njit
def can_move_numba(all_x, all_y, all_w, all_h, types, idx, direction, num_blocks):
    block_type = types[idx]
    dx = 1 if direction == 1 and block_type == 0 else (-1 if direction == 0 and block_type == 0 else 0)
    dy = 1 if direction == 1 and block_type == 1 else (-1 if direction == 0 and block_type == 1 else 0)
    return check_collision_numba(all_x[idx], all_y[idx], all_w[idx], all_h[idx],
                                dx, dy, all_x, all_y, all_w, all_h, idx, num_blocks)

# ----------------- NUMBA ACTION MASK -----------------
@njit
def compute_action_mask(all_x, all_y, all_w, all_h, types, num_blocks):
    mask = np.zeros(MAX_BLOCKS * 2, dtype=np.int8)
    for idx in range(num_blocks):
        for direction in (0, 1):
            if can_move_numba(all_x, all_y, all_w, all_h, types, idx, direction, num_blocks):
                mask[idx * 2 + direction] = 1
    return mask

# ----------------- NUMBA GET_OBS -----------------
@njit
def get_obs_numba(all_x, all_y, all_w, all_h, types, is_key_flags, num_blocks):
    obs = np.zeros((MAX_BLOCKS, FEATURES_PER_BLOCK), dtype=np.float32)
    for i in range(num_blocks):
        obs[i, 0] = all_x[i] / (GRID_SIZE - 1)
        obs[i, 1] = all_y[i] / (GRID_SIZE - 1)
        obs[i, 2] = all_w[i] / GRID_SIZE
        obs[i, 3] = all_h[i] / GRID_SIZE
        obs[i, 4] = types[i]   # 0 = H, 1 = V
        obs[i, 5] = is_key_flags[i]
    return obs.flatten()


class PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, text_level=None, max_steps=50):
        super().__init__()
        self.width = GRID_SIZE
        self.height = GRID_SIZE
        self.max_steps = max_steps

        self.text_level = text_level
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.step_num = 0
        self.total_steps = 0
        self.last_action = None
        self.prev_actions = []
        self.prev_key_x = 0
        self.logging_enabled = True
        self.log_step_counter = 0

        # ----------------- Векторные массивы для Numba -----------------
        self.all_x = np.full(MAX_BLOCKS, -1, dtype=np.int32)
        self.all_y = np.full(MAX_BLOCKS, -1, dtype=np.int32)
        self.all_w = np.zeros(MAX_BLOCKS, dtype=np.int32)
        self.all_h = np.zeros(MAX_BLOCKS, dtype=np.int32)
        self.types = np.zeros(MAX_BLOCKS, dtype=np.int8)   # 0=H,1=V
        self.is_key_flags = np.zeros(MAX_BLOCKS, dtype=np.int8)
        self.num_blocks = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_BLOCKS * 2)

    def set_logging_enabled(self, enabled: bool):
        self.logging_enabled = enabled

    # ----------------- MASK FOR MaskablePPO -----------------
    def action_mask(self):
        return compute_action_mask(self.all_x, self.all_y, self.all_w,
                                   self.all_h, self.types, self.num_blocks)

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.last_action = None
        self.prev_actions = []
        self.prev_key_x = 0
        self.step_num = 0
        self.log_step_counter = 0

        if self.text_level is None:
            self.blocks, self.block_texts, self.key_id = generate_default_level()
        else:
            self.blocks, self.block_texts, self.key_id = parse_level(text_level=self.text_level)

        # ----------------- обновление векторных массивов -----------------
        self.num_blocks = len(self.blocks)
        self.all_x.fill(-1)
        self.all_y.fill(-1)
        self.all_w.fill(0)
        self.all_h.fill(0)
        self.types.fill(0)
        self.is_key_flags.fill(0)
        for i, (bid, b) in enumerate(self.blocks.items()):
            self.all_x[i] = b["x"]
            self.all_y[i] = b["y"]
            self.all_w[i] = b["w"]
            self.all_h[i] = b["h"]
            self.types[i] = 0 if b["type"] == "H" else 1
            self.is_key_flags[i] = 1 if bid == self.key_id else 0

        if self.logging_enabled:
            log_level(self, self.text_level)

        return self._get_obs(), {}

    # ----------------- observation -----------------
    def _get_obs(self):
        return get_obs_numba(self.all_x, self.all_y, self.all_w,
                             self.all_h, self.types, self.is_key_flags, self.num_blocks)

    # ----------------- helpers -----------------
    def _index_to_block_id(self, chosen_index):
        block_items = list(self.blocks.items())
        if 0 <= chosen_index < len(block_items):
            return block_items[chosen_index][0]
        return None

    def _can_move(self, block_id, direction):
        idx = list(self.blocks.keys()).index(block_id)
        return can_move_numba(self.all_x, self.all_y, self.all_w, self.all_h,
                              self.types, idx, direction, self.num_blocks)

    # ----------------- move -----------------
    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"] == "H":
            block["x"] += 1 if direction == 1 else -1
        else:
            block["y"] += 1 if direction == 1 else -1

        # обновление векторных массивов
        idx = list(self.blocks.keys()).index(block_id)
        self.all_x[idx] = block["x"]
        self.all_y[idx] = block["y"]
        self.all_w[idx] = block["w"]
        self.all_h[idx] = block["h"]

    # ----------------- reward -----------------
    def _compute_reward(self, block_id, direction, moved, violated, is_reverse, is_solved, terminated, invalid_action, is_key_moved):
        reward = 0
        if moved:
            reward -= 0.05
        if invalid_action:
            reward -= 1.0
        if self.last_action is not None and self.last_action[0] == block_id and self.last_action[1] == direction:
            reward += 0.03
        if violated:
            reward -= 3.0
        if is_reverse:
            reward -= 3.0
        if is_key_moved:
            key = self.blocks[self.key_id]
            reward += key["x"] * 0.5
        if is_solved:
            reward += 10.0
        if not is_solved and terminated:
            reward -= 5.0
        return reward

    # ----------------- is_solved -----------------
    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"] + key["w"] - 1 == (self.width - 1)

    # ----------------- step -----------------
    # ----------------- step -----------------
    def step(self, action):
        if self.logging_enabled:
            log_action_mask(self, action, self.log_step_counter)

        action = int(action)
        chosen_index = action // 2
        direction = action % 2

        real_block_id = self._index_to_block_id(chosen_index)
        invalid_action = real_block_id is None

        moved = False
        violated = False
        is_reverse = False

        prev_key_x = self.prev_key_x

        # для подсветки предыдущей позиции блока
        prev_pos = None
        arrow = None

        if not invalid_action:
            is_reverse = (
                    self.last_action is not None and
                    self.last_action[0] == real_block_id and
                    self.last_action[1] != direction
            )
            if self._can_move(real_block_id, direction):
                # сохраняем старую позицию перед движением
                block = self.blocks[real_block_id]
                if block["type"] == "H":
                    arrow = "⬅" if direction == 0 else "⮕"
                    # если движемся влево, подсвечиваем правый край блока
                    prev_x = block["x"] + block["w"] - 1 if direction == 0 else block["x"]
                    prev_pos = (prev_x, block["y"])
                else:
                    arrow = "⬆" if direction == 0 else "⬇"
                    # если движемся вверх, подсвечиваем нижний край блока
                    prev_y = block["y"] + block["h"] - 1 if direction == 0 else block["y"]
                    prev_pos = (block["x"], prev_y)

                self._move_block(real_block_id, direction)
                moved = True
            else:
                violated = True

        is_key_moved = False
        key = self.blocks[self.key_id]
        if key["x"] > prev_key_x:
            is_key_moved = True
            self.prev_key_x = key["x"]

        is_solved = self._is_solved()
        terminated = is_solved or self.step_num >= self.max_steps - 1
        truncated = False

        reward = self._compute_reward(
            real_block_id if real_block_id is not None else -1,
            direction, moved, violated, is_reverse, is_solved, terminated, invalid_action, is_key_moved
        )

        self.prev_actions.append(f"{real_block_id}:{direction}")

        if self.logging_enabled:
            log_action(action, self, moved, reward, highlight_from=prev_pos, highlight_dir=arrow)
            self.log_step_counter += 1

        self.step_num += 1
        self.total_steps += 1
        self.last_action = (real_block_id, direction) if not invalid_action else None

        obs = self._get_obs()
        info = {"is_success": is_solved}
        return obs, reward, terminated, truncated, info


