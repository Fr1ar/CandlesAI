RESET = "\033[0m"

FG_BLACK = "\033[30m"

# Яркие фоны
BG_BLACK_BRIGHT  = "\033[100m"
BG_GREEN_BRIGHT  = "\033[102m"
BG_RED_BRIGHT    = "\033[101m"
BG_YELLOW_BRIGHT = "\033[103m"


def make_cell(bg, char):
    return f"{bg}{FG_BLACK} {char} {RESET}"


def render_pretty_colored(env):
    empty = make_cell(BG_BLACK_BRIGHT, " ")

    grid = [[empty for _ in range(6)] for _ in range(6)]

    for block_id, block in env.blocks.items():
        x, y, w, h, t = block["x"], block["y"], block["w"], block["h"], block["type"]

        # ID всех блоков черным
        if block_id == env.key_id:
            char = "0"
            bg = BG_YELLOW_BRIGHT
        else:
            char = env.block_texts.get(block_id, "?")
            bg = BG_GREEN_BRIGHT if t == "H" else BG_RED_BRIGHT

        cell = make_cell(bg, char)

        # заполняем блок на сетке
        for dy in range(h):
            for dx in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 6 and 0 <= nx < 6:
                    grid[ny][nx] = cell

    # выводим ровно без пробелов
    for row in grid:
        print("".join(row))


def action_to_text(action, env):
    block_id, direction = action
    block = env.blocks.get(block_id)
    if block is None:
        return f"Блок {block_id} не найден"

    char = "0" if block_id == env.key_id else env.block_texts.get(block_id, "?")
    type_str = (
        "ключ" if block_id == env.key_id
        else "горизонтальная свеча" if block["type"] == "H"
        else "вертикальная свеча"
    )

    if block["type"] == "H":
        dir_str = "влево" if direction == 0 else "вправо"
    else:
        dir_str = "вверх" if direction == 0 else "вниз"

    return f"Двигаем {type_str} '{char}' {dir_str}"


def log_action(action, env, moved, step, reward):
    if not moved:
        print(f"\nШаг {step + 1}, {action_to_text(action, env)}, блок не сдвинулся, штраф {reward}")
    else:
        print(f"\nШаг {step + 1}, {action_to_text(action, env)}, награда: {reward}")

    render_pretty_colored(env)
