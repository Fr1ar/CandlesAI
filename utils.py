import numpy as np

from arrows import LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW

RESET = "\033[0m"
FG_BLACK = "\033[30m"
FG_WHITE = "\033[97m"

# Яркие фоны
BG_BLACK_BRIGHT = "\033[40m"
BG_GREEN_BRIGHT = "\033[102m"
BG_RED_BRIGHT = "\033[101m"
BG_YELLOW_BRIGHT = "\033[103m"

FG_GRAY_DARK = "\033[90m"


def make_cell(bg, char, fg_color=FG_BLACK):
    # Центрируем символ в клетке шириной 3
    return f"{bg}{fg_color}{char:^3}{RESET}"


def render_pretty_colored(env, prev_block_pos=None, direction=None):
    empty = make_cell(BG_BLACK_BRIGHT, " ")
    grid = [[empty for _ in range(6)] for _ in range(6)]

    # рисуем все блоки
    for block_id, block in env.blocks.items():
        x, y, w, h, t = block["x"], block["y"], block["w"], block["h"], block["type"]

        if block_id == env.key_id:
            char = "0"
            bg = BG_YELLOW_BRIGHT
        else:
            char = env.block_texts.get(block_id, "?")
            bg = BG_GREEN_BRIGHT if t == "H" else BG_RED_BRIGHT

        for dy in range(h):
            for dx in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 6 and 0 <= nx < 6:
                    grid[ny][nx] = make_cell(bg, char)

    # рисуем подсветку после всех блоков
    if prev_block_pos is not None:
        if prev_block_pos["type"] == "H":
            # если движемся влево, подсвечиваем правый край блока
            if direction == 0:
                prev_direction = "⏴"
                prev_x = prev_block_pos["x"] + prev_block_pos["w"] - 1
            else:
                prev_direction = "⏵"
                prev_x = prev_block_pos["x"]
            prev_pos = (prev_x, prev_block_pos["y"])
        else:
            # если движемся вверх, подсвечиваем нижний край блока
            if direction == 0:
                prev_direction = "▲"
                prev_y = prev_block_pos["y"] + prev_block_pos["h"] - 1
            else:
                prev_direction = "▼"
                prev_y = prev_block_pos["y"]
            prev_pos = (prev_block_pos["x"], prev_y)

        hx, hy = prev_pos
        if 0 <= hy < 6 and 0 <= hx < 6:
            grid[hy][hx] = make_cell(BG_BLACK_BRIGHT, prev_direction, fg_color=FG_WHITE)

    # рамка
    top_border = f"{FG_GRAY_DARK}┌" + "───" * 6 + "┐" + RESET
    bottom_border = f"{FG_GRAY_DARK}└" + "───" * 6 + "┘" + RESET
    print(top_border)
    for row in grid:
        print(f"{FG_GRAY_DARK}│{RESET}" + "".join(row) + f"{FG_GRAY_DARK}│{RESET}")
    print(bottom_border)


def action_to_text(action, env):
    if isinstance(action, (list, tuple)):
        block_index = int(action[0])
        direction = int(action[1]) if len(action) > 1 else 0
    elif hasattr(action, "dtype") and isinstance(action, (np.ndarray,)):
        a = int(np.asarray(action).flatten()[0])
        block_index = a // 2
        direction = a % 2
    else:
        a = int(action)
        block_index = a // 2
        direction = a % 2

    block_items = list(env.blocks.items())
    if not (0 <= block_index < len(block_items)):
        dir_str = "влево/вверх" if direction == 0 else "вправо/вниз"
        return f"Некорректное действие: block_index {block_index} отсутствует, направление {dir_str}"

    block_id = block_items[block_index][0]
    block = env.blocks.get(block_id)
    if block is None:
        return f"Блок {block_id} не найден"

    char = "0" if block_id == env.key_id else env.block_texts.get(block_id, "?")
    type_str = "ключ" if block_id == env.key_id else "блок"

    if block["type"] == "H":
        dir_str = LEFT_ARROW if direction == 0 else RIGHT_ARROW
    else:
        dir_str = UP_ARROW if direction == 0 else DOWN_ARROW

    return f"Двигаем {type_str} '{char}' {dir_str} "


def log_action_mask(env, step, total_steps):
    step_fmt = f"{step + 1:,}"
    total_steps_fmt = f"{total_steps + 1:,}"

    print(f"Шаг {step_fmt} (всего: {total_steps_fmt})")

    mask = env.action_mask()
    block_ids = list(env.blocks.keys())

    for block_idx, block_id in enumerate(block_ids):
        allowed_dirs = []

        block_type = env.blocks[block_id]["type"]

        for direction in (0, 1):
            action_index = block_idx * 2 + direction
            if mask[action_index]:
                if block_type == "H":
                    allowed_dirs.append(LEFT_ARROW if direction == 0 else RIGHT_ARROW)
                else:
                    allowed_dirs.append(UP_ARROW if direction == 0 else DOWN_ARROW)

        if allowed_dirs:
            block_name = env.block_texts.get(block_id, str(block_id))
            print(f" • Блок '{block_name}' можно двигать: {' '.join(allowed_dirs)}")


def log_action(action, env, moved, reward, cur_dist, prev_block_pos=None, direction=None):
    action_text = action_to_text(action, env)
    if not moved:
        print(f"{action_text} блок не сдвинулся, штраф {reward:.2f}")
    else:
        print(f"{action_text}, ходов до решения: {cur_dist} (награда: {reward:.2f})")

    render_pretty_colored(env, prev_block_pos, direction)
    print("-" * 40)


def log_level(env, text_level, cur_dist):
    print(f'Уровень: "{text_level}"')
    print(f'Шагов до решения: {cur_dist}')
    render_pretty_colored(env)
    print("-" * 40)
