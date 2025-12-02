from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnv
import json


class SequentialMultiLevelEnv(PuzzleEnv):
    def __init__(self, levels):
        self.levels = levels
        self.level_index = 0
        super().__init__(exploration_prob=0.3, text_level=None)

    def reset(self, seed=None, options=None):
        self.text_level = self.levels[self.level_index]
        self.level_index = (self.level_index + 1) % len(self.levels)
        return super().reset(seed=seed, options=options)


if __name__ == "__main__":
    # загрузка уровней
    levels_file = "levels/single.json"
    with open(levels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    levels = data.get("levels", [])
    if not levels:
        raise ValueError("Файл пуст")

    def make_env_func():
        return SequentialMultiLevelEnv(levels)

    env = make_vec_env(make_env_func, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.1,
        n_epochs=10,
    )

    model.learn(total_timesteps=500_000)

    model.save("puzzle_ppo_explore_model")
    print("Обучение завершено.")
