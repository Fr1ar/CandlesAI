import numpy as np

RESET = "\033[0m"

FG_BLACK = "\033[30m"

# Яркие фоны
BG_BLACK_BRIGHT  = "\033[40m"
BG_GREEN_BRIGHT  = "\033[102m"
BG_RED_BRIGHT    = "\033[101m"
BG_YELLOW_BRIGHT = "\033[103m"

FG_GRAY_DARK = "\033[90m"

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

        for dy in range(h):
            for dx in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 6 and 0 <= nx < 6:
                    grid[ny][nx] = cell

    # --- рамка тёмно-серого цвета ---
    top_border = f"{FG_GRAY_DARK}┌" + "───" * 6 + "┐" + RESET
    bottom_border = f"{FG_GRAY_DARK}└" + "───" * 6 + "┘" + RESET
    print(top_border)
    for row in grid:
        print(f"{FG_GRAY_DARK}│{RESET}" + "".join(row) + f"{FG_GRAY_DARK}│{RESET}")
    print(bottom_border)

def action_to_text(action, env):
    """
    Accepts action in several forms:
      - scalar int (Discrete action)
      - tuple/list (block_index, direction)  -- kept for compatibility
      - numpy array
    """
    # normalize to (block_index, direction) and produce human-readable text
    if isinstance(action, (list, tuple)):
        # if user passed tuple (block_idx, dir)
        block_index = int(action[0])
        direction = int(action[1]) if len(action) > 1 else 0
    elif hasattr(action, "dtype") and isinstance(action, (np.ndarray,)):
        a = int(np.asarray(action).flatten()[0])
        block_index = a // 2
        direction = a % 2
    else:
        # scalar
        a = int(action)
        block_index = a // 2
        direction = a % 2

    # map block_index -> block_id (real)
    block_items = list(env.blocks.items())
    if not (0 <= block_index < len(block_items)):
        dir_str = ("влево/вверх" if direction == 0 else "вправо/вниз")
        return f"Некорректное действие: block_index {block_index} отсутствует, направление {dir_str}"

    block_id = block_items[block_index][0]
    block = env.blocks.get(block_id)
    if block is None:
        return f"Блок {block_id} не найден"

    char = "0" if block_id == env.key_id else env.block_texts.get(block_id, "?")
    type_str = ("ключ" if block_id == env.key_id else "блок")

    if block["type"] == "H":
        dir_str = "влево" if direction == 0 else "вправо"
    else:
        dir_str = "вверх" if direction == 0 else "вниз"

    return f"Двигаем {type_str} '{char}' {dir_str}"


def log_action_mask(env, step, total_steps):
    step_fmt = f"{step + 1:,}"
    total_steps_fmt = f"{total_steps + 1:,}"

    print(f"Шаг {step_fmt} (всего: {total_steps_fmt})")

    mask = env.action_mask()
    block_ids = list(env.blocks.keys())

    for block_idx, block_id in enumerate(block_ids):
        allowed_dirs = []

        block_type = env.blocks[block_id]["type"]  # "H" или "V"

        for direction in (0, 1):
            action_index = block_idx * 2 + direction
            if mask[action_index]:
                if block_type == "H":
                    allowed_dirs.append("⬅" if direction == 0 else "⮕")
                else:
                    allowed_dirs.append("⬆" if direction == 0 else "⬇")

        if allowed_dirs:
            block_name = env.block_texts.get(block_id, str(block_id))
            print(f' • Блок "{block_name}" можно двигать: {" ".join(allowed_dirs)}')


def log_action(action, env, moved, step, total_steps, reward):
    action_text = action_to_text(action, env)
    log_action_mask(env, step, total_steps)

    if not moved:
        print(f"{action_text}, блок не сдвинулся, штраф {reward:.2f}")
    else:
        print(f"{action_text}, награда: {reward:.2f}")

    render_pretty_colored(env)
    print("-" * 40)

def log_level(env, text_level):
    print(f"УРОВЕНЬ: \"{text_level}\"")
    render_pretty_colored(env)
    print("-" * 40)
