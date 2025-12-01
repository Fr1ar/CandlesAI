from stable_baselines3 import PPO
from env import PuzzleEnvExplore
from utils import log_action, render_pretty_colored
import time

def is_solvable_debug(text_level, model_path="puzzle_ppo_explore_model", max_steps=200, delay=0.2):
    env = PuzzleEnvExplore(text_level=text_level)
    obs, _ = env.reset()
    model = PPO.load(model_path)

    print("=== Начальный уровень ===")
    render_pretty_colored(env)
    print("\n")
    time.sleep(delay)

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        moved = info.get("moved", False)

        log_action(action, env, moved, step, reward)

        print("-"*30)
        time.sleep(delay)

        if terminated:
            print("\nУровень пройден!")
            return True

    print("\nУровень непроходимый по версии модели")
    return False


if __name__ == "__main__":
    # Пример текстового уровня
    text_level = "- - b c c c.a a b - - -.- 0 0 d - -.f f f d e e.- - g g g x.z z - h h x"
    result = is_solvable_debug(text_level, max_steps=100_000, delay=0)
    print("Проходимость уровня:", result)
