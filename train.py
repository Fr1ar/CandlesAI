from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnv
import json


class SequentialMultiLevelEnv(PuzzleEnv):
    """Выбирает уровни строго по порядку, а не случайно"""
    def __init__(self, levels, exploration_prob=0.3):
        self.levels = levels
        self.level_index = 0
        super().__init__(exploration_prob=exploration_prob, text_level=None)

    def reset(self, seed=None, options=None):
        # выбираем уровень по порядку
        self.text_level = self.levels[self.level_index]

        # следующий reset возьмёт следующий уровень
        self.level_index = (self.level_index + 1) % len(self.levels)

        return super().reset(seed=seed, options=options)


if __name__ == "__main__":
    # Загрузка уровней
    levels_file = "levels/single.json"
    with open(levels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    levels = data.get("levels", [])
    if not levels:
        raise ValueError(f"'{levels_file}' не содержит уровней")

    def make_env_func():
        return SequentialMultiLevelEnv(levels, exploration_prob=0.3)

    env = make_vec_env(make_env_func, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.1,
        n_epochs=10,
    )

    model.learn(total_timesteps=200_000)

    model.save("puzzle_ppo_explore_model")
    print("Обучение завершено и модель сохранена.")
