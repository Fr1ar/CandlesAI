from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PuzzleEnv
from parser import load_levels

class SequentialMultiLevelEnv(PuzzleEnv):
    def __init__(self, levels):
        self.levels = levels
        self.level_index = 0
        super().__init__(text_level=None)

    def reset(self, seed=None, options=None):
        self.text_level = self.levels[self.level_index]
        self.level_index = (self.level_index + 1) % len(self.levels)
        return super().reset(seed=seed, options=options)

def run():
    levels = load_levels("levels/difficult.json")

    def make_env_func():
        return SequentialMultiLevelEnv(levels)

    env = make_vec_env(make_env_func, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.2,   # достаточно высокая энтропия, чтобы пробовать разные блоки
        n_epochs=10,
        # verbose=1,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("output/puzzle_model")
    print("Done")

if __name__ == "__main__":
    run()
