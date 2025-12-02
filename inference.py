from stable_baselines3 import PPO
from env import PuzzleEnv
from parser import load_levels
from utils import render_pretty_colored


def is_solvable_single(text_level, model, max_steps):
    env = PuzzleEnv(text_level=text_level, max_steps=max_steps)
    obs, _ = env.reset()

    print("=== Начальный уровень ===")
    render_pretty_colored(env)
    print()

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, is_solved, info = env.step(action)

        if is_solved:
            print("Уровень пройден!\n")
            return True

    print("Нет решения.\n")
    return False


def check_all_levels(levels, model_path="output/puzzle_model", max_steps=300):
    model = PPO.load(model_path)

    results = []
    for i, level in enumerate(levels):
        print(f"=== Уровень {i+1}/{len(levels)} ===")
        ok = is_solvable_single(level, model, max_steps=max_steps)
        results.append((i, ok))

    return results


def run():
    levels = load_levels("levels/single.json")
    results = check_all_levels(levels, max_steps=200)

    print("=== Итоги ===")
    for idx, ok in results:
        print(f"Уровень {idx}: {'✅ проходим' if ok else '❌ не проходим'}")


if __name__ == "__main__":
    run()
