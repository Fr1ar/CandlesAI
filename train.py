from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import os
import glob
import random
import numpy as np

from env import PuzzleEnv
from parser import load_levels

# ----------------- ПАРАМЕТРЫ -----------------
# количество параллельных сред
n_envs = 16
# Сколько всего шагов
total_timesteps = 10_000_000_000
# Через сколько шагов делать чекпойнт
checkpoint_freq = 100_000_000
# Как часто выводить в лог количество шагов
log_every_n_timesteps = 100_000
# Через сколько шагов увеличивать сложность
min_moves_increment_timesteps = 10_000_000
# Начальная сложность
current_min_moves = 0
# Максимальная сложность
max_min_moves = 30
# Шанс, что агенту попадётся простой уровень
simple_level_chance = 0.2

final_model = "puzzle_model"
final_model_file = f"{final_model}.zip"
final_model_path = f"output/{final_model_file}"
checkpoint_pattern = f"output/{final_model}_*.zip"
levels_path = "levels/dataset.json"


# ----------------------------------------------
def log(text):
    now = datetime.now()
    print(f"{now.strftime('%H:%M:%S')} {text}")


# ----------------- CALLBACK ДЛЯ ПЕРИОДИЧЕСКОГО СОХРАНЕНИЯ -----------------
class SaveEveryNStepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0
        self.last_log_timestep = 0

    def _on_step(self) -> bool:
        if 0 < log_every_n_timesteps < self.num_timesteps - self.last_log_timestep:
            self._log(
                f"Шаг = {self.num_timesteps + 1:,}, (сложность = {current_min_moves})"
            )
            self.last_log_timestep = self.num_timesteps

        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            path = f"{self.save_path}_{self.num_timesteps}.zip"
            self.model.save(path)

            # Сохраняем текущее состояние обучения
            self._save_training_state(path)

            self._log(f"Модель сохранена в {path}")
        return True

    def _log(self, text):
        if self.verbose:
            log(text)

    def _save_training_state(self, path):
        import zipfile
        import json

        data = {
            "current_min_moves": current_min_moves,
        }
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("training_state.json", json.dumps(data))


# ----------------- MULTI-LEVEL ENV -----------------
class SequentialMultiLevelEnv(PuzzleEnv):
    def __init__(self, levels, step_increment=min_moves_increment_timesteps):
        super().__init__(text_level=None)
        self.levels = levels
        self.step_increment = step_increment
        self.total_steps_done = 0
        self.logging_enabled = True

        # --- numpy массив для min_moves всех уровней ---
        self.min_moves_array = np.array(
            [lvl.get("meta", {}).get("min_moves", 0) for lvl in levels]
        )
        self.max_min_moves = self.min_moves_array.max()
        self.num_levels = len(levels)

    def reset(self, seed=None, options=None):
        global current_min_moves, simple_level_chance

        # Уровни проще текущей сложности
        simple_mask = self.min_moves_array <= current_min_moves
        simple_indices = np.flatnonzero(simple_mask)

        # Уровни сложнее текущей сложности
        difficult_mask = self.min_moves_array <= current_min_moves
        difficult_indices = np.flatnonzero(difficult_mask)

        # Если почему-то ничего не подошло
        if simple_indices.size == 0:
            simple_indices = np.arange(self.num_levels)
        if difficult_indices.size == 0:
            difficult_indices = np.arange(self.num_levels)

        # Берём простой уровень с шансом simple_level_chance
        if simple_indices.size > 0 and random.random() < simple_level_chance:
            chosen_idx = random.choice(simple_indices)
            level_type = "легкий"
        else:
            chosen_idx = random.choice(difficult_indices)
            level_type = "сложный"

        current_level = self.levels[chosen_idx]
        self.text_level = current_level["data"]

        meta = current_level.get("meta", {})
        min_moves = meta.get("min_moves", 0)

        if self.logging_enabled:
            log(f"Сложность уровня: {min_moves}, тип: {level_type}")

        # --- Постепенное увеличение сложности ---
        self.total_steps_done += self.step_num
        if (
            self.total_steps_done >= self.step_increment
            and current_min_moves < self.max_min_moves
            and current_min_moves <= max_min_moves
        ):
            self.total_steps_done = 0
            current_min_moves += 1
            if self.logging_enabled:
                log(f"Увеличиваем сложность: {current_min_moves}")

        return super().reset(seed=seed, options=options)


# ----------------- ФУНКЦИИ -----------------
def make_env_func(levels):
    env = SequentialMultiLevelEnv(levels)
    env.set_logging_enabled(n_envs == 1)
    return ActionMasker(env, lambda env: env.action_mask())


def delete_checkpoints():
    files = glob.glob(checkpoint_pattern)
    for f in files:
        os.remove(f)
    if files:
        log(f"Удалено {len(files)} старых чекпойнтов")


def get_checkpoint_files():
    return sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)


# ----------------- ОСНОВНАЯ ФУНКЦИЯ -----------------
def run():
    global current_min_moves
    log("Начало тренировки...")
    os.makedirs("output", exist_ok=True)

    levels = load_levels(levels_path)
    env = make_vec_env(lambda: make_env_func(levels), n_envs=n_envs)

    resume = False
    model = None
    checkpoint_files = get_checkpoint_files()

    # ----- Проверка финальной модели -----
    if os.path.exists(final_model_path):
        answer = (
            input(
                "Найдена финальная модель puzzle_model.zip. "
                "Хотите её перезаписать? При согласии финальная модель и все чекпойнты будут удалены. [y/N]: "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            os.remove(final_model_path)
            delete_checkpoints()
            log("Финальная модель и чекпойнты удалены. Начинаем обучение с нуля.")
        else:
            log("Финальная модель не будет перезаписана. Выход.")
            return

    # ----- Проверка чекпойнтов, если финальной модели нет -----
    elif checkpoint_files:
        answer = (
            input(
                "Найдена незаконченная модель (чекпойнты). "
                "Хотите продолжить обучение с последнего чекпойнта? "
                "Если нет, старые чекпойнты будут удалены. [y/N]: "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            latest_checkpoint = checkpoint_files[-1]
            log(f"Загружаем {latest_checkpoint} для продолжения обучения...")
            model = MaskablePPO.load(latest_checkpoint)
            model.set_env(env)
            resume = True

            # --- Загрузка training_state.json ---
            import zipfile
            import json

            try:
                with zipfile.ZipFile(latest_checkpoint, "r") as archive:
                    if "training_state.json" in archive.namelist():
                        data = json.loads(archive.read("training_state.json"))
                        current_min_moves = data.get("current_min_moves", 0)

                        log(
                            f"Восстановлено состояние: current_min_moves={current_min_moves}"
                        )
            except Exception as e:
                log(f"Не удалось восстановить training_state.json: {e}")

        else:
            delete_checkpoints()
            log("Старые чекпойнты удалены. Начинаем обучение с нуля.")
            resume = False

    # ----- Создаем новую модель, если не продолжаем -----
    if not resume:
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

    # ----- Callback для периодического сохранения -----
    callback = SaveEveryNStepsCallback(
        save_freq=checkpoint_freq, save_path="output/puzzle_model"
    )

    # ----- Обучение -----
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=not resume,
    )

    # ----- Сохранение финальной модели -----
    model.save(final_model_path)
    log(f"Тренировка завершена. Финальная модель сохранена в {final_model_file}")


if __name__ == "__main__":
    run()
