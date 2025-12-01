from stable_baselines3 import PPO
from env import PuzzleEnvText
from utils import log_action, render_pretty_colored
import time

def is_solvable_debug(text_level, model_path="puzzle-rl-text-model", max_steps=200, delay=0.2):
    env = PuzzleEnvText()
    obs, _ = env.reset(text_level=text_level)
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
    text_level = "- - b c c c.a a b - - -.- 0 0 d - -.f f f d e e.- - g g g x.z z - h h x"
    result = is_solvable_debug(text_level)
    print("Проходимость уровня:", result)
