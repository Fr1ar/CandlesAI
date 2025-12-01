import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PuzzleEnvText(gym.Env):
    """
    Среда "свечи и ключ" с поддержкой text_level.
    Проверка столкновений и штраф за недвигаемые блоки.
    Добавлена маска действий для RL.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.width = 6
        self.height = 6

        self.blocks = {}       # block_id -> {"x","y","w","h","type"}
        self.key_id = None
        self.state = None
        self.block_texts = {}

        self.action_space = spaces.MultiDiscrete([1,2])
        self.observation_space = spaces.Box(low=0, high=255, shape=(6,6), dtype=np.uint8)

    def reset(self, text_level=None, seed=None, options=None):
        self.blocks = {}
        self.block_texts = {}
        self.key_id = None
        next_id = 0

        if text_level is None:
            # стандартный тестовый уровень
            self.blocks[next_id] = {"x":0, "y":0, "w":2, "h":1, "type":"H"}  # ключ
            self.key_id = next_id
            self.block_texts[next_id] = "0"
            next_id += 1

            self.blocks[next_id] = {"x":2, "y":0, "w":3, "h":1, "type":"H"}  # горизонтальная свеча
            self.block_texts[next_id] = "a"
            next_id += 1

            self.blocks[next_id] = {"x":0, "y":1, "w":1, "h":3, "type":"V"}  # вертикальная свеча
            self.block_texts[next_id] = "b"
            next_id += 1

        else:
            # Разбор text_level
            lines = text_level.split(".")
            for y, line in enumerate(lines):
                cells = line.split()
                x = 0
                while x < len(cells):
                    cell = cells[x]
                    if cell != "-":
                        w = 1
                        h = 1
                        while x + w < len(cells) and cells[x + w] == cell:
                            w += 1
                        if cell == "0":
                            block_type = "H"
                            self.key_id = next_id
                        else:
                            block_type = "H" if w > 1 else "V"
                        self.blocks[next_id] = {"x":x, "y":y, "w":w, "h":h, "type":block_type}
                        self.block_texts[next_id] = cell
                        next_id += 1
                        x += w
                    else:
                        x += 1

        if self.key_id is None:
            raise ValueError("В text_level не найден ключ '0' и стандартный уровень не создан")

        self.action_space = spaces.MultiDiscrete([len(self.blocks), 2])
        self.state = self._get_obs()
        return self.state, {}

    def _get_obs(self):
        obs = np.zeros((6,6), dtype=np.uint8)
        for block_id, block in self.blocks.items():
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            for dy in range(h):
                for dx in range(w):
                    if 0 <= y+dy < 6 and 0 <= x+dx < 6:
                        obs[y+dy, x+dx] = block_id + 1
        return obs

    def _can_move(self, block_id, direction):
        if block_id not in self.blocks:
            return False
        block = self.blocks[block_id]
        dx, dy = 0, 0
        if block["type"]=="H":
            dx = -1 if direction==0 else 1
        else:
            dy = -1 if direction==0 else 1

        # проверка границ
        if block["x"]+dx < 0 or block["x"]+block["w"]+dx > self.width:
            return False
        if block["y"]+dy < 0 or block["y"]+block["h"]+dy > self.height:
            return False

        # проверка пересечения с другими блоками
        for other_id, other in self.blocks.items():
            if other_id == block_id:
                continue
            if (block["x"]+dx < other["x"]+other["w"] and
                    block["x"]+block["w"]+dx > other["x"] and
                    block["y"]+dy < other["y"]+other["h"] and
                    block["y"]+block["h"]+dy > other["y"]):
                return False

        return True

    def _move_block(self, block_id, direction):
        block = self.blocks[block_id]
        if block["type"]=="H":
            if direction==0:
                block["x"] = max(0, block["x"]-1)
            else:
                block["x"] = min(self.width-block["w"], block["x"]+1)
        else:
            if direction==0:
                block["y"] = max(0, block["y"]-1)
            else:
                block["y"] = min(self.height-block["h"], block["y"]+1)

    def _compute_reward(self):
        return 0.0

    def _is_solved(self):
        key = self.blocks[self.key_id]
        return key["x"] + key["w"] - 1 == 5

    def get_action_mask(self):
        """
        Возвращает маску доступных действий для RL:
        shape = [num_blocks, 2], True если действие доступно
        """
        mask = np.zeros((len(self.blocks), 2), dtype=np.bool_)
        for block_id in self.blocks:
            for direction in [0,1]:
                if self._can_move(block_id, direction):
                    mask[block_id, direction] = True
        return mask

    def step(self, action):
        block_id, direction = action
        reward = 0.0
        terminated = False
        truncated = False
        moved = False

        if self._can_move(block_id, direction):
            self._move_block(block_id, direction)
            moved = True
            reward = self._compute_reward()
        else:
            moved = False
            reward = -0.1  # усиленный штраф за попытку недвигаемого блока

        if self._is_solved():
            terminated = True
            reward += 1.0

        obs = self._get_obs()
        info = {"moved": moved, "action_mask": self.get_action_mask()}
        return obs, reward, terminated, truncated, info
