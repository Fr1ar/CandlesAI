import random, json, string
from collections import deque, namedtuple
import os

from levels.arrows import LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW

WIDTH = HEIGHT = 6
KEY_ID = "0"
KEY_Y = 2
KEY_LEN = 2
BLOCK_MIN = 2
BLOCK_MAX = 4

Block = namedtuple("Block", ["id", "orient", "length", "x", "y"])

# ------------------------ Helpers ------------------------


def place_blocks(blocks):
    grid = [["-" for _ in range(WIDTH)] for __ in range(HEIGHT)]
    for b in blocks:
        if b.orient == "H":
            for dx in range(b.length):
                grid[b.y][b.x + dx] = b.id
        else:
            for dy in range(b.length):
                grid[b.y + dy][b.x] = b.id
    return grid


def grid_to_string(grid):
    return ".".join(" ".join(r) for r in grid)


def occupied_grid(blocks):
    grid = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for b in blocks:
        if b.orient == "H":
            for dx in range(b.length):
                grid[b.y][b.x + dx] = b.id
        else:
            for dy in range(b.length):
                grid[b.y + dy][b.x] = b.id
    return grid


def neighbors_single_step_with_move(blocks):
    grid = occupied_grid(blocks)
    result = []
    for i, b in enumerate(blocks):
        if b.orient == "H":
            if b.x - 1 >= 0 and grid[b.y][b.x - 1] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x - 1, b.y)
                result.append((new, (b.id, LEFT_ARROW)))
            if b.x + b.length < WIDTH and grid[b.y][b.x + b.length] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x + 1, b.y)
                result.append((new, (b.id, RIGHT_ARROW)))
        else:
            if b.y - 1 >= 0 and grid[b.y - 1][b.x] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x, b.y - 1)
                result.append((new, (b.id, UP_ARROW)))
            if b.y + b.length < HEIGHT and grid[b.y + b.length][b.x] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x, b.y + 1)
                result.append((new, (b.id, DOWN_ARROW)))
    return result


# ------------------------ Безопасная генерация блоков ------------------------


def generate_random_blocks_safe(min_h, max_h, min_v, max_v, min_blockers):
    letters = list(string.ascii_lowercase)
    random.shuffle(letters)
    blocks = []
    letter_i = 0

    key_x_pos = random.choice([0, 1])
    blocks.append(Block(KEY_ID, "H", KEY_LEN, key_x_pos, KEY_Y))
    used = {(key_x_pos + i, KEY_Y) for i in range(KEY_LEN)}

    def place_block(orient):
        nonlocal letter_i
        if letter_i >= len(letters):
            return None
        lid = letters[letter_i]
        letter_i += 1
        length = random.randint(BLOCK_MIN, BLOCK_MAX)
        for attempt in range(50):
            if orient == "H":
                y = random.randint(0, HEIGHT - 1)
                x = random.randint(0, WIDTH - length)
                if all((x + dx, y) not in used for dx in range(length)):
                    for dx in range(length):
                        used.add((x + dx, y))
                    return Block(lid, "H", length, x, y)
            else:
                x = random.randint(0, WIDTH - 1)
                y = random.randint(0, HEIGHT - length)
                if all((x, y + dy) not in used for dy in range(length)):
                    for dy in range(length):
                        used.add((x, y + dy))
                    return Block(lid, "V", length, x, y)
        return None

    n_h = random.randint(min_h, max_h)
    placed_h = 0
    while placed_h < n_h:
        b = place_block("H")
        if b:
            blocks.append(b)
            placed_h += 1
        else:
            n_h -= 1
            if n_h < placed_h:
                break

    n_v = random.randint(min_v, max_v)
    placed_v = 0
    while placed_v < n_v:
        b = place_block("V")
        if b:
            blocks.append(b)
            placed_v += 1
        else:
            n_v -= 1
            if n_v < placed_v:
                break

    blockers_count = 0
    key_line_x_end = key_x_pos + KEY_LEN
    attempts_blockers = 0
    max_attempts_blockers = 100
    while blockers_count < min_blockers and attempts_blockers < max_attempts_blockers:
        attempts_blockers += 1
        if letter_i >= len(letters):
            break
        lid = letters[letter_i]
        letter_i += 1
        length = random.randint(BLOCK_MIN, BLOCK_MAX)
        y = KEY_Y
        max_x = WIDTH - length
        if max_x < key_line_x_end:
            continue
        x = random.randint(key_line_x_end, max_x)
        if all((x + dx, y) not in used for dx in range(length)):
            for dx in range(length):
                used.add((x + dx, y))
            blocks.append(Block(lid, "H", length, x, y))
            blockers_count += 1

    return blocks


# ------------------------ BFS решатель ------------------------


def solve_with_moves(blocks):
    start_state = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in blocks))
    visited = set([start_state])
    queue = deque([(blocks, [])])

    while queue:
        cur_blocks, moves = queue.popleft()
        key = next(b for b in cur_blocks if b.id == KEY_ID)
        if key.x + KEY_LEN - 1 == WIDTH - 1 and key.y == KEY_Y:
            return moves, len(moves)

        for neigh_blocks, (bid, symbol) in neighbors_single_step_with_move(cur_blocks):
            state = tuple(
                sorted((b.id, b.orient, b.length, b.x, b.y) for b in neigh_blocks)
            )
            if state in visited:
                continue
            visited.add(state)
            queue.append((neigh_blocks, moves + [(bid, symbol)]))

    return None, None


# ------------------------ Метаданные (компактный вариант) ------------------------


def build_meta_compact(blocks, moves):
    horiz_sizes = [b.length for b in blocks if b.id != KEY_ID and b.orient == "H"]
    vert_sizes = [b.length for b in blocks if b.id != KEY_ID and b.orient == "V"]
    key = next(b for b in blocks if b.id == KEY_ID)
    moves_list = [f"{bid}{symbol}" for (bid, symbol) in moves]
    return {
        "h_blocks": horiz_sizes,
        "v_blocks": vert_sizes,
        "key_x": key.x,
        "moves": moves_list,
    }


# ------------------------ JSON с объектами с отступом и массивами в одну строку ------------------------


def _dump_compact_arrays(obj, f, indent=2):
    """Сериализация: dict с отступами, list в одну строку"""

    def _serialize(o, level=0):
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                items.append(
                    " " * (level * indent)
                    + json.dumps(k)
                    + ": "
                    + _serialize(v, level + 1)
                )
            pad = " " * ((level - 1) * indent) if level > 0 else ""
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        elif isinstance(o, list):
            return "[" + ",".join(_serialize(x, level + 1) for x in o) + "]"
        else:
            return json.dumps(o, ensure_ascii=False)

    f.write(_serialize(obj))


def save_json(obj, file_path, indent=2, use_standard_json=False):
    if use_standard_json:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            _dump_compact_arrays(obj, f, indent=indent)


# ------------------------ Массовая генерация ------------------------


def generate_levels(settings, file_path, use_standard_json=False):
    levels_list = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if (
                    isinstance(existing, dict)
                    and "levels" in existing
                    and isinstance(existing["levels"], list)
                ):
                    levels_list = existing["levels"]
        except Exception:
            levels_list = []

    for s in settings:
        name = s["name"]
        target = s["count"]
        print(f"Generating {target} levels for '{name}'...")
        generated = 0
        attempts = 0
        while generated < target:
            attempts += 1
            blocks = generate_random_blocks_safe(
                s["min_h"], s["max_h"], s["min_v"], s["max_v"], s.get("min_blockers", 0)
            )
            moves, steps = solve_with_moves(blocks)
            if moves is None:
                continue
            if steps is None or steps < s.get("min_steps", 0):
                continue

            level_data = grid_to_string(place_blocks(blocks))
            meta = build_meta_compact(blocks, moves)

            entry = {"data": level_data, "meta": meta}
            levels_list.append(entry)
            generated += 1

            # запись JSON с возможностью выбора стандартного формата
            save_json(
                {"levels": levels_list},
                file_path,
                indent=2,
                use_standard_json=use_standard_json,
            )

            print(
                f"  ✅ Level {generated} for '{name}' generated successfully (min steps: {steps})"
            )
        print(f"Finished '{name}' in {attempts} attempts.")
    return {"levels": levels_list}


# ------------------------ Пример использования ------------------------

def run():
    file_path = "levels/generated_auto_with_meta.json"
    count = 10
    settings = [
        {
            "name": "elementary",
            "min_h": 1,
            "max_h": 2,
            "min_v": 1,
            "max_v": 2,
            "min_blockers": 1,
            "min_steps": 3,
            "count": count,
        },
        {
            "name": "easy",
            "min_h": 2,
            "max_h": 3,
            "min_v": 2,
            "max_v": 3,
            "min_blockers": 1,
            "min_steps": 5,
            "count": count,
        },
        {
            "name": "medium",
            "min_h": 3,
            "max_h": 4,
            "min_v": 3,
            "max_v": 4,
            "min_blockers": 2,
            "min_steps": 7,
            "count": count,
        },
        {
            "name": "hard",
            "min_h": 4,
            "max_h": 5,
            "min_v": 4,
            "max_v": 5,
            "min_blockers": 3,
            "min_steps": 10,
            "count": count,
        },
        {
            "name": "very_hard",
            "min_h": 5,
            "max_h": 6,
            "min_v": 5,
            "max_v": 6,
            "min_blockers": 4,
            "min_steps": 12,
            "count": count,
        },
    ]

    # use_standard_json=False => компактный формат по умолчанию
    all_levels = generate_levels(settings, file_path, use_standard_json=False)
    print(f"✅ Finished. Saved to '{file_path}'")


if __name__ == "__main__":
    run()
