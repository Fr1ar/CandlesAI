import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils import log_action
import random

class PuzzleEnvExplore(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, exploration_prob=0.3):
        super().__init__()
        self.width = 6
        self.height = 6
        self.stepNum = 0
        self.blocks = {}
        self.key_id = None
        self.state = None
        self.block_texts = {}
        self.exploration_prob = exploration_prob  # вероятность случайного действия

        # action: (block_id, direction)
        self.action_space = spaces.MultiDiscrete([1,2])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6*6*3,), dtype=np.float32)

    def generate_default_level(self):
        next_id = 0
        self.blocks[next_id] = {"x": 0, "y": 2, "w": 2, "h": 1, "type": "H"}  # ключ
        self.key_id = next_id
        self.block_texts[next_id] = "0"
        next_id += 1
        self.blocks[next_id] = {"x": 2, "y": 0, "w": 3, "h": 1, "type": "H"}  # горизонтальная свеча
        self.block_texts[next_id] = "a"
        next_id += 1
        self.blocks[next_id] = {"x": 3, "y": 1, "w": 1, "h": 3, "type": "V"}  # вертикальная свеча
        self.block_texts[next_id] = "b"
        next_id += 1

    def reset(self, text_level=None, seed=None, options=None):
        super().reset(seed=seed)
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        self.stepNum = 0
        self.generate_default_level()
        self.state = self._get_obs()
        return self.state, {}

    def _get_obs(self):
        obs = np.zeros((6,6,3), dtype=np.float32)
        for block_id, block in self.blocks.items():
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            channel = 0 if block_id==self.key_id else 1 if block["type"]=="H" else 2
            for dy in range(h):
                for dx in range(w):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < 6 and 0 <= nx < 6:
                        obs[ny,nx,channel] = 1.0
        return obs.flatten()

    def _can_move(self, block_id, direction):
        if block_id not in self.blocks:
            return False
        block = self.blocks[block_id]
        dx, dy = 0, 0
        if block["type"]=="H": dx = -1 if direction==0 else 1
        else: dy = -1 if direction==0 else 1
        if block["x"]+dx < 0 or block["x"]+block["w"]+dx>self.width: return False
        if block["y"]+dy <0 or block["y"]+block["h"]+dy>self.height: return False
        for other_id, other in self.blocks.items():
            if other_id==block_id: continue
            if (block["x"]+dx < other["x"]+other["w"] and
                block["x"]+block["w"]+dx > other["x"] and
                block["y"]+dy < other["y"]+other["h"] and
                block["y"]+block["h"]+dy > other["y"]):
                return False
        return True

    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"]=="H":
            block["x"] = max(0, min(self.width-block["w"], block["x"] + (1 if direction==1 else -1)))
        else:
            block["y"] = max(0, min(self.height-block["h"], block["y"] + (1 if direction==1 else -1)))

    def _compute_reward(self, block_id, moved):
        reward = 0.0
        key = self.blocks[self.key_id]
        reward += (key["x"]+key["w"]-1)*0.01  # небольшой бонус за продвижение ключа
        if moved and block_id != self.key_id:
            reward += 0.2  # бонус за движение свечей
        return reward

    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"]+key["w"]-1 == 5

    def step(self, action):
        # --- forced random exploration ---
        if np.random.rand() < self.exploration_prob:
            num_blocks = len(self.blocks)
            block_id = random.randint(0, num_blocks-1)
            direction = random.randint(0, 1)
            action = (block_id, direction)

        block_id, direction = action
        moved = False
        if self._can_move(block_id, direction):
            self._move_block(block_id, direction)
            moved = True
        reward = self._compute_reward(block_id, moved)
        terminated = self._is_solved()
        truncated = False # self.stepNum >= 1000000
        info = {"is_success": terminated}
        # log_action(action, self, moved, self.stepNum, reward)
        if terminated:
            print('Выход найден за {0} ходов'.format(self.stepNum))

        self.stepNum += 1
        return self._get_obs(), reward, terminated, truncated, info
