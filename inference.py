from sb3_contrib import MaskablePPO
from env import PuzzleEnv
from parser import load_levels

def is_solvable_single(text_level, model, max_steps=200):
    env = PuzzleEnv(text_level=text_level, max_steps=max_steps)
    obs, _ = env.reset()

    for step in range(max_steps):
        # MaskablePPO требует маску при predict
        mask = env.action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("is_success") or env._is_solved():
            print("Уровень пройден!\n")
            return True

    print("Нет решения.\n")
    return False


def check_all_levels(levels, model_path="output/puzzle_model", max_steps=200):
    model = MaskablePPO.load(model_path)

    results = []
    for i, level in enumerate(levels):
        print(f"=== Уровень {i+1}/{len(levels)} ===")
        ok = is_solvable_single(level, model, max_steps=max_steps)
        results.append((i, ok))

    return results


def run():
    levels = load_levels("levels/test.json")
    results = check_all_levels(levels, max_steps=200)

    print("=== Итоги ===")
    for idx, ok in results:
        print(f"Уровень {idx}: {'✅ проходим' if ok else '❌ не проходим'}")


if __name__ == "__main__":
    run()
