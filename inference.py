from stable_baselines3 import PPO
from env import PuzzleEnvText
import time

RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"

def render_pretty_colored(env):
    grid = [["-" for _ in range(6)] for _ in range(6)]
    for block_id, block in env.blocks.items():
        x, y, w, h, t = block["x"], block["y"], block["w"], block["h"], block["type"]
        if block_id == env.key_id:
            char = "0"
            color = YELLOW
        else:
            char = env.block_texts.get(block_id, "?")
            color = GREEN if t=="H" else RED
        for dy in range(h):
            for dx in range(w):
                ny, nx = y+dy, x+dx
                if 0 <= ny < 6 and 0 <= nx < 6:
                    grid[ny][nx] = f"{color}{char}{RESET}"
    for row in grid:
        print(" ".join(row))

def action_to_text(action, env):
    block_id, direction = action
    block = env.blocks.get(block_id)
    if block is None:
        return f"Блок {block_id} не найден"
    char = "0" if block_id==env.key_id else env.block_texts.get(block_id, "?")
    type_str = "ключ" if block_id==env.key_id else "горизонтальная свеча" if block["type"]=="H" else "вертикальная свеча"
    if block["type"]=="H":
        dir_str = "влево" if direction==0 else "вправо"
    else:
        dir_str = "вверх" if direction==0 else "вниз"
    return f"Двигаем {type_str} '{char}' {dir_str}"

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

        if not moved:
            print(f"\nШаг {step+1}, {action_to_text(action, env)}, блок не сдвинулся, штраф {reward}")
        else:
            print(f"\nШаг {step+1}, {action_to_text(action, env)}, награда: {reward}")

        render_pretty_colored(env)
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
