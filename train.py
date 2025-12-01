from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnvExplore

if __name__ == "__main__":
    # создаём векторизованную среду
    text_level = "- - b c c c.a a b - - -.- 0 0 d - -.f f f d e e.- - g g g x.z z - h h x"
    puzzle = PuzzleEnvExplore(exploration_prob=0.3, text_level=text_level)
    env = make_vec_env(lambda: puzzle, n_envs=1)

    # создаём модель PPO
    model = PPO(
        "MlpPolicy",
        env,
        # verbose=2,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.1,  # высокая энтропия для better exploration
        n_epochs=10
    )

    # обучение
    model.learn(total_timesteps=200_000)

    # сохраняем модель
    model.save("puzzle_ppo_explore_model")
    print("Обучение завершено и модель сохранена.")
