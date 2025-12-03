from stable_baselines3 import PPO
from env import PuzzleEnv
from parser import load_levels
from utils import render_pretty_colored


def is_solvable_single(text_level, model, max_steps=200):
    env = PuzzleEnv(text_level=text_level, max_steps=max_steps)
    obs, _ = env.reset()

    print("=== Начальный уровень ===")
    render_pretty_colored(env)
    print()

    for step in range(max_steps):
        # MultiDiscrete action: model.predict возвращает 1D int array
        action, _ = model.predict(obs, deterministic=True)
        # если action MultiDiscrete, то берем tuple для step
        if hasattr(env.action_space, "nvec"):
            action = tuple(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("is_success") or env._is_solved():
            print("Уровень пройден!\n")
            return True

    print("Нет решения.\n")
    return False


def check_all_levels(levels, model_path="output/puzzle_model", max_steps=200):
    model = PPO.load(model_path)

    results = []
    for i, level in enumerate(levels):
        print(f"=== Уровень {i+1}/{len(levels)} ===")
        ok = is_solvable_single(level, model, max_steps=max_steps)
        results.append((i, ok))

    return results


def run():
    levels = load_levels("levels/difficult.json")
    results = check_all_levels(levels, max_steps=200)

    print("=== Итоги ===")
    for idx, ok in results:
        print(f"Уровень {idx}: {'✅ проходим' if ok else '❌ не проходим'}")


if __name__ == "__main__":
    run()
