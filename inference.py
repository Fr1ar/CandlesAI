from stable_baselines3 import PPO
from env import PuzzleEnv
from parser import load_levels
from utils import render_pretty_colored


def is_solvable_single(text_level, model, max_steps=200, delay=0.0):
    env = PuzzleEnv(text_level=text_level)
    obs, _ = env.reset()

    print("=== Начальный уровень ===")
    render_pretty_colored(env)
    print()

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print("Уровень пройден!\n")
            return True

    print("Нет решения.\n")
    return False


def check_all_levels(levels, model_path="puzzle_ppo_explore_model", max_steps=300):
    model = PPO.load(model_path)

    results = []
    for i, level in enumerate(levels):
        print(f"=== Уровень {i+1}/{len(levels)} ===")
        ok = is_solvable_single(level, model, max_steps=max_steps)
        results.append((i, ok))

    return results


def run():
    levels = load_levels("levels/single.json")
    results = check_all_levels(levels, max_steps=100_000)

    print("=== Итоги ===")
    for idx, ok in results:
        print(f"Уровень {idx}: {'✓ проходим' if ok else '✗ не проходим'}")


if __name__ == "__main__":
    run()
