import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parser import parse_level, generate_default_level
from utils import log_action

# Максимум блоков (свечи+ключ)
MAX_BLOCKS = 15

# Сколько числовых признаков на блок (x,y,w,h,type_H,is_key) -> 6
FEATURES_PER_BLOCK = 6

# Размер матрицы наблюдений
OBS_SIZE = MAX_BLOCKS * FEATURES_PER_BLOCK

class PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, text_level=None, max_steps=50):
        super().__init__()
        self.width = 6
        self.height = 6
        self.max_steps = max_steps

        self.text_level = text_level

        # состояние уровня
        self.blocks = {}         # dict: id -> {x,y,w,h,type}
        self.block_texts = {}
        self.key_id = None
        self.step_num = 0
        self.total_steps = 0
        self.last_key_x = 0
        self.last_action = None

        # --- FIXED spaces (must be set in __init__) ---
        # flattened observation: MAX_BLOCKS * FEATURES_PER_BLOCK
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # MultiDiscrete: choose block index [0..MAX_BLOCKS-1] and direction [0..1]
        self.action_space = spaces.MultiDiscrete([MAX_BLOCKS, 2])

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.last_action = None
        self.step_num = 0

        # parse or generate
        if self.text_level is None:
            self.blocks, self.block_texts, self.key_id = generate_default_level()
        else:
            self.blocks, self.block_texts, self.key_id = parse_level(text_level=self.text_level)

        # prepare last_key_x
        key = self.blocks[self.key_id]
        self.last_key_x = key["x"]

        obs = self._get_obs()
        return obs, {}

    # ----------------- observation (fixed-size flattened) -----------------
    def _get_obs(self):
        arr = np.zeros((MAX_BLOCKS, FEATURES_PER_BLOCK), dtype=np.float32)
        # fill existing blocks in order of their ids (assuming contiguous small ids)
        # If block ids are sparse, we map them into 0..(n-1) by enumerating items
        for i, (block_id, block) in enumerate(self.blocks.items()):
            if i >= MAX_BLOCKS:
                break
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            type_H = 1.0 if block["type"] == "H" else 0.0
            is_key = 1.0 if block_id == self.key_id else 0.0
            # normalize positions/sizes by grid size
            arr[i, 0] = x / (self.width - 1)   # normalized x in [0,1]
            arr[i, 1] = y / (self.height - 1)
            arr[i, 2] = w / self.width
            arr[i, 3] = h / self.height
            arr[i, 4] = type_H
            arr[i, 5] = is_key
        return arr.flatten()

    # ----------------- helper: map chosen index -> real block id -----------------
    def _index_to_block_id(self, chosen_index):
        # if there are fewer blocks than MAX_BLOCKS, chosen_index may be invalid
        # map used indices 0..n-1 to actual block ids (enumeration order)
        block_items = list(self.blocks.items())
        if chosen_index < 0:
            return None
        if chosen_index < len(block_items):
            return block_items[chosen_index][0]  # actual block_id
        return None  # invalid (no block at that slot)

    # ----------------- can move / move -----------------
    def _can_move(self, block_id, direction):
        block = self.blocks[block_id]
        dx = (1 if direction == 1 else -1) if block["type"] == "H" else 0
        dy = (1 if direction == 1 else -1) if block["type"] == "V" else 0
        # border
        if block["x"] + dx < 0 or block["x"] + block["w"] + dx > self.width:
            return False
        if block["y"] + dy < 0 or block["y"] + block["h"] + dy > self.height:
            return False
        # collisions
        for oid, other in self.blocks.items():
            if oid == block_id:
                continue
            if (block["x"] + dx < other["x"] + other["w"] and
                block["x"] + block["w"] + dx > other["x"] and
                block["y"] + dy < other["y"] + other["h"] and
                block["y"] + block["h"] + dy > other["y"]):
                return False
        return True

    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"] == "H":
            block["x"] += 1 if direction == 1 else -1
        else:
            block["y"] += 1 if direction == 1 else -1

    # ----------------- reward -----------------
    def _compute_reward(self, block_id, moved, violated, is_reverse, is_solved, invalid_action=False):
        # baseline small negative step so agent prefers shorter solutions
        reward = -0.05

        if invalid_action:
            # small penalty for selecting empty slot / nonexistent block
            reward -= 0.3

        key = self.blocks[self.key_id]
        if moved and block_id == self.key_id and key["x"] > self.last_key_x:
            self.last_key_x = key["x"]
            reward += 1.0  # stronger positive for key progress

        if moved and block_id != self.key_id:
            reward += 0.2  # small positive for moving other blocks (encourage exploring them)

        if violated:
            reward -= 0.6

        if is_reverse:
            reward -= 0.4

        if is_solved:
            reward += 10.0

        if not is_solved and self.step_num >= self.max_steps:
            reward -= 5.0

        return reward

    # ----------------- solved -----------------
    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"] + key["w"] - 1 == (self.width - 1)

    # ----------------- step -----------------
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.flatten()
        # when action comes as e.g. array([a,b]) or scalar - handle both
        try:
            chosen_index = int(action[0])
            direction = int(action[1])
        except Exception:
            # fallback: scalar -> treat as Discrete(MAX_BLOCKS*2) style
            a = int(action)
            chosen_index = a // 2
            direction = a % 2

        # map chosen index to real block id (if exists)
        real_block_id = self._index_to_block_id(chosen_index)
        invalid_action = real_block_id is None

        is_reverse = False
        moved = False
        violated = False

        if not invalid_action:
            is_reverse = (
                self.last_action is not None
                and self.last_action[0] == real_block_id
                and self.last_action[1] != direction
            )
            if self._can_move(real_block_id, direction):
                self._move_block(real_block_id, direction)
                moved = True
            else:
                violated = True

        is_solved = self._is_solved()
        terminated = is_solved or self.step_num >= self.max_steps
        truncated = False

        reward = self._compute_reward(
            real_block_id if real_block_id is not None else -1,
            moved,
            violated,
            is_reverse,
            is_solved,
            invalid_action=invalid_action,
        )

        log_action(action, self, moved, self.step_num, self.total_steps, reward)

        self.last_action = (real_block_id, direction) if not invalid_action else None
        self.step_num += 1
        self.total_steps += 1

        obs = self._get_obs()
        info = {"is_success": is_solved}
        return obs, reward, terminated, truncated, info