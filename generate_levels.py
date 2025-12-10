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
    """
    Возвращает список (new_blocks, (block_id, dir_symbol))
    """
    grid = occupied_grid(blocks)
    result = []
    for i, b in enumerate(blocks):
        if b.orient == "H":
            # left
            if b.x - 1 >= 0 and grid[b.y][b.x - 1] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x - 1, b.y)
                result.append((new, (b.id, LEFT_ARROW)))
            # right
            if b.x + b.length < WIDTH and grid[b.y][b.x + b.length] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x + 1, b.y)
                result.append((new, (b.id, RIGHT_ARROW)))
        else:
            # up
            if b.y - 1 >= 0 and grid[b.y - 1][b.x] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x, b.y - 1)
                result.append((new, (b.id, UP_ARROW)))
            # down
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

    # ключ на старте
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

    # горизонтальные блоки
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

    # вертикальные блоки
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

    # блоки на линии ключа для гарантии min_blockers
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


# ------------------------ BFS решатель: возвращает путь (moves) ------------------------


def solve_with_moves(blocks):
    """
    BFS, возвращает список ходов (как list of (id, symbol)),
    и также возвращает конечное количество шагов (len of moves).
    Если нерешаем — возвращает (None, None)
    """
    start_state = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in blocks))
    visited = set([start_state])
    queue = deque([(blocks, [])])  # (blocks_list, moves_list)

    while queue:
        cur_blocks, moves = queue.popleft()
        # check goal
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


# ------------------------ Метаданные ------------------------


def build_meta(blocks, moves):
    # horizontal and vertical summary excluding key
    horiz_sizes = {}
    vert_sizes = {}
    horiz_total = 0
    vert_total = 0
    for b in blocks:
        if b.id == KEY_ID:
            continue
        if b.orient == "H":
            horiz_total += 1
            horiz_sizes[b.length] = horiz_sizes.get(b.length, 0) + 1
        else:
            vert_total += 1
            vert_sizes[b.length] = vert_sizes.get(b.length, 0) + 1

    # convert sizes keys to strings (like example)
    horiz_sizes_str = {str(k): v for k, v in horiz_sizes.items()}
    vert_sizes_str = {str(k): v for k, v in vert_sizes.items()}

    # starting key_x (find key in initial blocks)
    key = next(b for b in blocks if b.id == KEY_ID)
    key_x = key.x

    # moves to required format: list of single-key dicts
    moves_list = [{str(bid): symbol} for (bid, symbol) in moves]

    meta = {
        "horizontal_blocks": {"total": horiz_total, "sizes": horiz_sizes_str},
        "vertical_blocks": {"total": vert_total, "sizes": vert_sizes_str},
        "key_x": key_x,
        "moves": moves_list,
    }
    return meta


# ------------------------ Массовая генерация — теперь с новым json-форматом ------------------------


def generate_levels(settings, file_path):
    # load existing (if any) to continue
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
            # ignore parse errors and start fresh
            levels_list = []

    # count how many already generated per category (not strictly needed, we just append)
    # We'll still print per-category progress based on len of appended items for that run.
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
            # try solve and get moves
            moves, steps = solve_with_moves(blocks)
            if moves is None:
                continue
            # check min_steps if present
            min_steps_required = s.get("min_steps", 0)
            if steps is None or steps < min_steps_required:
                continue

            # build entry
            level_data = grid_to_string(place_blocks(blocks))
            meta = build_meta(blocks, moves)

            entry = {"data": level_data, "meta": meta}
            levels_list.append(entry)
            generated += 1

            # write to disk after each successful generation (update top-level structure)
            out = {"levels": levels_list}
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            # log
            print(
                f"  ✅ Level {generated} for '{name}' generated successfully (min steps: {steps})"
            )
        print(f"Finished '{name}' in {attempts} attempts.")
    return {"levels": levels_list}


# ------------------------ Пример использования ------------------------

if __name__ == "__main__":
    file_path = "levels/generated_auto_with_meta.json"
    N = 10  # количество уровней на категорию
    settings = [
        {
            "name": "elementary",
            "min_h": 1,
            "max_h": 2,
            "min_v": 1,
            "max_v": 2,
            "min_blockers": 1,
            "min_steps": 3,
            "count": N,
        },
        {
            "name": "easy",
            "min_h": 2,
            "max_h": 3,
            "min_v": 2,
            "max_v": 3,
            "min_blockers": 1,
            "min_steps": 5,
            "count": N,
        },
        {
            "name": "medium",
            "min_h": 3,
            "max_h": 4,
            "min_v": 3,
            "max_v": 4,
            "min_blockers": 2,
            "min_steps": 7,
            "count": N,
        },
        {
            "name": "hard",
            "min_h": 4,
            "max_h": 5,
            "min_v": 4,
            "max_v": 5,
            "min_blockers": 3,
            "min_steps": 10,
            "count": N,
        },
        {
            "name": "very_hard",
            "min_h": 5,
            "max_h": 6,
            "min_v": 5,
            "max_v": 6,
            "min_blockers": 4,
            "min_steps": 12,
            "count": N,
        },
    ]

    all_levels = generate_levels(settings, file_path)

    print(f"✅ Finished. Saved to '{file_path}'")
