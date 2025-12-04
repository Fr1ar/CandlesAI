from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os

from env import PuzzleEnv
from parser import load_levels


# ------------------------------------------
# CALLBACK ДЛЯ ПЕРИОДИЧЕСКОГО СОХРАНЕНИЯ
# ------------------------------------------
class SaveEveryNStepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            path = f"{self.save_path}_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose:
                print(f"{current_time} [Callback] Model saved to {path}")
        return True


# ------------------------------------------
# MULTI-LEVEL ENV
# ------------------------------------------
class SequentialMultiLevelEnv(PuzzleEnv):
    def __init__(self, levels):
        self.levels = levels
        self.level_index = 0
        super().__init__(text_level=None)

    def reset(self, seed=None, options=None):
        self.text_level = self.levels[self.level_index]
        self.level_index = (self.level_index + 1) % len(self.levels)
        return super().reset(seed=seed, options=options)


# ------------------------------------------
# ФУНКЦИЯ СОЗДАНИЯ СРЕДЫ С MASKER
# ------------------------------------------
def make_env_func(levels):
    env = SequentialMultiLevelEnv(levels)
    env.set_logging_enabled(False)
    return ActionMasker(env, lambda env: env.action_mask())


# ------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# ------------------------------------------
def run(total_timesteps=1_000_000, checkpoint_freq=100_000, resume=False):
    print("Training...")
    levels = load_levels("levels/difficult.json")

    # Создаем векторную среду
    env = make_vec_env(lambda: make_env_func(levels), n_envs=1)

    # Проверяем, есть ли сохранённая модель
    model_path = "output/puzzle_model.zip"
    if resume and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = MaskablePPO.load(model_path)
        model.set_env(env)
        reset_timesteps = False
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.2,
            n_epochs=10,
        )
        reset_timesteps = True

    # Callback для периодического сохранения
    callback = SaveEveryNStepsCallback(save_freq=checkpoint_freq, save_path="output/puzzle_model")

    # Обучаем / дообучаем
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_timesteps)

    # Сохраняем финальную модель
    model.save("output/puzzle_model")
    print("Training done. Final model saved as puzzle_model.zip")


if __name__ == "__main__":
    run()
