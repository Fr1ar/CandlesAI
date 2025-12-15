import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parser import parse_level
from utils import log_action, log_level, log_action_mask
from numba import njit
from collections import deque

MAX_BLOCKS = 15
FEATURES_PER_BLOCK = 6
OBS_SIZE = MAX_BLOCKS * FEATURES_PER_BLOCK
GRID_SIZE = 6


# ----------------- STATE ENCODING -----------------
def encode_state(all_x, all_y, num_blocks):
    return tuple(all_x[:num_blocks]) + tuple(all_y[:num_blocks])


# ----------------- NUMBA COLLISION CHECK -----------------
@njit
def check_collision_numba(
    x, y, w, h, dx, dy, all_x, all_y, all_w, all_h, block_idx, num_blocks
):
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
        if (
            x + dx < ox + ow
            and x + w + dx > ox
            and y + dy < oy + oh
            and y + h + dy > oy
        ):
            return False
    return True


# ----------------- NUMBA CAN_MOVE -----------------
@njit
def can_move_numba(all_x, all_y, all_w, all_h, types, idx, direction, num_blocks):
    block_type = types[idx]
    dx = (
        1
        if direction == 1 and block_type == 0
        else (-1 if direction == 0 and block_type == 0 else 0)
    )
    dy = (
        1
        if direction == 1 and block_type == 1
        else (-1 if direction == 0 and block_type == 1 else 0)
    )
    return check_collision_numba(
        all_x[idx],
        all_y[idx],
        all_w[idx],
        all_h[idx],
        dx,
        dy,
        all_x,
        all_y,
        all_w,
        all_h,
        idx,
        num_blocks,
    )


# ----------------- NUMBA ACTION MASK -----------------
@njit
def compute_action_mask(all_x, all_y, all_w, all_h, types, num_blocks):
    mask = np.zeros(MAX_BLOCKS * 2, dtype=np.int8)
    for idx in range(num_blocks):
        for direction in (0, 1):
            if can_move_numba(
                all_x, all_y, all_w, all_h, types, idx, direction, num_blocks
            ):
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
        obs[i, 4] = types[i]
        obs[i, 5] = is_key_flags[i]
    return obs.flatten()


# ----------------- NUMBA is_solved -----------------
@njit
def is_solved_numba(all_x, all_w, key_index):
    return all_x[key_index] + all_w[key_index] - 1 == GRID_SIZE - 1


# ----------------- NUMBA reverse/invalid -----------------
@njit
def check_action_flags_numba(last_block_idx, last_dir, current_block_idx, current_dir):
    invalid = 1 if current_block_idx == -1 else 0
    reverse = 0
    if invalid == 0:
        if last_block_idx == current_block_idx and last_dir != current_dir:
            reverse = 1
    return invalid, reverse


# ----------------- NUMBA reward -----------------
@njit
def compute_reward_numba(
    block_id,
    direction,
    moved,
    violated,
    reverse,
    solved,
    terminated,
    invalid,
    key_moved,
    key_x,
    step_to_solution,
):
    reward = 0.0
    if moved:
        reward -= 0.05
    if invalid:
        reward -= 1.0
    if reverse:
        reward -= 3.0
    if violated:
        reward -= 3.0
    if key_moved:
        reward += key_x * 0.5
    if step_to_solution != 0:
        reward += step_to_solution * 3
    if solved:
        reward += 10.0
    if (not solved) and terminated:
        reward -= 5.0
    return reward


# ======================= BFS SOLVER =======================
@njit
def apply_move_numba(all_x, all_y, idx, direction, types):
    if types[idx] == 0:
        all_x[idx] += 1 if direction == 1 else -1
    else:
        all_y[idx] += 1 if direction == 1 else -1


def solve_level_bfs(env):
    num_blocks = env.num_blocks
    key_idx = list(env.blocks.keys()).index(env.key_id)

    start_x = env.all_x.copy()
    start_y = env.all_y.copy()

    visited = set()
    queue = deque()
    queue.append((start_x, start_y, 0))
    visited.add(encode_state(start_x, start_y, num_blocks))

    while queue:
        all_x, all_y, steps = queue.popleft()

        if is_solved_numba(all_x, env.all_w, key_idx):
            return steps  # возвращаем количество шагов до решения

        for idx in range(num_blocks):
            for direction in (0, 1):
                if not can_move_numba(
                    all_x,
                    all_y,
                    env.all_w,
                    env.all_h,
                    env.types,
                    idx,
                    direction,
                    num_blocks,
                ):
                    continue

                nx = all_x.copy()
                ny = all_y.copy()
                apply_move_numba(nx, ny, idx, direction, env.types)

                state = encode_state(nx, ny, num_blocks)
                if state not in visited:
                    visited.add(state)
                    queue.append((nx, ny, steps + 1))

    return 0  # если решение не найдено


# ========================================================================
# ===========================   ENV CLASS   ===============================
# ========================================================================


class PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, text_level=None, max_steps=80):
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
        self.prev_key_x = 0
        self.logging_enabled = True
        self.dist_to_solution = 0
        self.prev_dist_to_solution = None

        self.all_x = np.full(MAX_BLOCKS, -1, dtype=np.int32)
        self.all_y = np.full(MAX_BLOCKS, -1, dtype=np.int32)
        self.all_w = np.zeros(MAX_BLOCKS, dtype=np.int32)
        self.all_h = np.zeros(MAX_BLOCKS, dtype=np.int32)
        self.types = np.zeros(MAX_BLOCKS, dtype=np.int8)
        self.is_key_flags = np.zeros(MAX_BLOCKS, dtype=np.int8)
        self.num_blocks = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_BLOCKS * 2)

    def set_logging_enabled(self, enabled: bool):
        self.logging_enabled = enabled

    # ----------------- action mask -----------------
    def action_mask(self):
        return compute_action_mask(
            self.all_x, self.all_y, self.all_w, self.all_h, self.types, self.num_blocks
        )

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.last_action = None
        self.step_num = 0

        self.blocks, self.block_texts, self.key_id = parse_level(
            text_level=self.text_level
        )

        key = self.blocks[self.key_id]
        self.prev_key_x = key["x"]

        # update vector arrays
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

        # ----------------- считаем BFS сразу -----------------
        self.dist_to_solution = solve_level_bfs(self)
        self.prev_dist_to_solution = self.dist_to_solution

        if self.logging_enabled:
            log_level(self, self.text_level, self.prev_dist_to_solution)

        return self._get_obs(), {}

    # ----------------- obs -----------------
    def _get_obs(self):
        return get_obs_numba(
            self.all_x,
            self.all_y,
            self.all_w,
            self.all_h,
            self.types,
            self.is_key_flags,
            self.num_blocks,
        )

    # ----------------- helpers -----------------
    def _index_to_block_id(self, chosen_index):
        block_items = list(self.blocks.items())
        if 0 <= chosen_index < len(block_items):
            return block_items[chosen_index][0]
        return None

    def _can_move(self, block_id, direction):
        idx = list(self.blocks.keys()).index(block_id)
        return can_move_numba(
            self.all_x,
            self.all_y,
            self.all_w,
            self.all_h,
            self.types,
            idx,
            direction,
            self.num_blocks,
        )

    # ----------------- move -----------------
    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"] == "H":
            block["x"] += 1 if direction == 1 else -1
        else:
            block["y"] += 1 if direction == 1 else -1

        idx = list(self.blocks.keys()).index(block_id)
        self.all_x[idx] = block["x"]
        self.all_y[idx] = block["y"]
        self.all_w[idx] = block["w"]
        self.all_h[idx] = block["h"]

    # ----------------- solved check -----------------
    def _is_solved(self):
        key_index = list(self.blocks.keys()).index(self.key_id)
        return bool(is_solved_numba(self.all_x, self.all_w, key_index))

    # ----------------- step -----------------
    def step(self, action):
        if self.logging_enabled:
            log_action_mask(self, self.step_num, self.total_steps)

        action = int(action)
        chosen_index = action // 2
        direction = action % 2

        real_block_id = self._index_to_block_id(chosen_index)

        # ----- new numba reverse/invalid -----
        if real_block_id is None:
            current_idx = -1
        else:
            current_idx = list(self.blocks.keys()).index(real_block_id)

        if self.last_action is None:
            last_idx = -1
            last_dir = -1
        else:
            last_idx = list(self.blocks.keys()).index(self.last_action[0])
            last_dir = self.last_action[1]

        invalid_int, reverse_int = check_action_flags_numba(
            last_idx, last_dir, current_idx, direction
        )

        invalid_action = bool(invalid_int)
        is_reverse = bool(reverse_int)

        # ----- moving logic -----
        moved = False
        violated = False
        prev_block_pos = None

        if not invalid_action:
            if self._can_move(real_block_id, direction):
                if self.logging_enabled:
                    prev_block_pos = dict(self.blocks[real_block_id])
                self._move_block(real_block_id, direction)
                moved = True
            else:
                violated = True

        # ----- key movement -----
        key = self.blocks[self.key_id]
        is_key_moved = key["x"] > self.prev_key_x
        if is_key_moved:
            self.prev_key_x = key["x"]

        # ----------------- пересчитываем BFS каждый шаг -----------------
        cur_dist = solve_level_bfs(self)

        # Приблизился ли к решению головоломки
        step_to_solution = 0
        if (
            self.prev_dist_to_solution is not None
            and cur_dist < self.prev_dist_to_solution
        ):
            step_to_solution = (
                self.dist_to_solution - cur_dist
            ) / self.dist_to_solution
            # print(f"step_to_solution: {step_to_solution:.2f}")
            self.prev_dist_to_solution = cur_dist

        # ----- solved check -----
        is_solved = self._is_solved()
        terminated = is_solved or self.step_num >= self.max_steps - 1
        truncated = False

        # ----- Numba reward -----
        reward = compute_reward_numba(
            current_idx,
            direction,
            moved,
            violated,
            is_reverse,
            is_solved,
            terminated,
            invalid_action,
            is_key_moved,
            key["x"],
            step_to_solution,
        )

        if self.logging_enabled:
            log_action(action, self, moved, reward, cur_dist, prev_block_pos, direction)

        self.step_num += 1
        self.total_steps += 1
        self.last_action = (real_block_id, direction) if not invalid_action else None

        obs = self._get_obs()
        info = {"is_success": is_solved}
        return obs, reward, terminated, truncated, info
