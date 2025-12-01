from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnvExplore

# создаём среду
env = make_vec_env(PuzzleEnvExplore, n_envs=1)

# создаём PPO
model = PPO(
    "MlpPolicy",
    env,
    # verbose=2,
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.1,   # увеличенная энтропия для exploration
    n_epochs=10
)

# обучение
model.learn(total_timesteps=200_000)
model.save("puzzle_ppo_explore_model")
