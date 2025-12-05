from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
import glob

from env import PuzzleEnv
from parser import load_levels

# ----------------- ПАРАМЕТРЫ -----------------
n_envs = 20  # количество параллельных сред (ядра CPU)
total_timesteps_default = 100_000_000
checkpoint_freq = total_timesteps_default / 10  # каждые N шагов сохранять модель

# ------------------------------------------
# CALLBACK ДЛЯ ПЕРИОДИЧЕСКОГО СОХРАНЕНИЯ
# ------------------------------------------
class SaveEveryNStepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0  # последний шаг, на котором был чекпойнт

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
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
    env.set_logging_enabled(n_envs == 1)  # логируем только если одна среда
    return ActionMasker(env, lambda env: env.action_mask())

# ------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# ------------------------------------------
def run(total_timesteps=total_timesteps_default, checkpoint_freq=checkpoint_freq):
    print("Начало тренировки...")
    os.makedirs("output", exist_ok=True)

    levels = load_levels("levels/generated.json")

    # Создаем векторную среду с несколькими процессами
    env = make_vec_env(lambda: make_env_func(levels), n_envs=n_envs)

    final_model_path = "output/puzzle_model.zip"
    checkpoint_files = sorted(glob.glob("output/puzzle_model_*.zip"), key=os.path.getmtime)

    resume = False

    # Если есть чекпойнты, но нет финальной модели
    if checkpoint_files and not os.path.exists(final_model_path):
        print("Найдена незаконченная модель (чекпойнты), финальная модель отсутствует.")
        answer = input("Хотите продолжить обучение с последнего чекпойнта? [y/N]: ").strip().lower()
        if answer == "y":
            latest_checkpoint = checkpoint_files[-1]
            print(f"Загружаем {latest_checkpoint} для продолжения обучения...")
            model = MaskablePPO.load(latest_checkpoint)
            model.set_env(env)
            resume = True

    if not resume:
        # Новая модель
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=n_envs * 64,
            learning_rate=2.5e-4,
            ent_coef=0.12,
            n_epochs=10,
            device="auto",
            verbose=(0 if n_envs == 1 else 1),
        )

    # Callback для периодического сохранения
    callback = SaveEveryNStepsCallback(save_freq=checkpoint_freq, save_path="output/puzzle_model")

    # Обучаем / дообучаем
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=not resume)

    # Сохраняем финальную модель
    model.save(final_model_path)
    print("Training done. Final model saved as puzzle_model.zip")


if __name__ == "__main__":
    run()
