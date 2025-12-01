from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnvText

if __name__ == "__main__":
    # Векторизуем среду для обучения
    env = make_vec_env(PuzzleEnvText, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        tensorboard_log="./tb/"
    )

    # Обучение
    model.learn(total_timesteps=100_000)

    # Сохраняем модель
    model.save("puzzle-rl-text-model")
    print("Модель обучена и сохранена")
