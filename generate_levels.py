import random, json, string
from collections import deque, namedtuple
import os

from arrows import LEFT_ARROW, RIGHT_ARROW, UP_ARROW, DOWN_ARROW

WIDTH = HEIGHT = 6
KEY_ID = "0"
KEY_Y = 2
KEY_LEN = 2
BLOCK_MIN = 2
BLOCK_MAX = 4
MAX_BFS_STEPS = 100

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


# ------------------------ BFS Solver ------------------------


# ------------------------ BFS Solver ------------------------

def solve_with_moves(blocks, max_steps=MAX_BFS_STEPS):
    """
    BFS с ограничением на максимальное количество ходов.
    Если количество ходов превысит max_steps, возвращает (None, None)
    """
    start_state = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in blocks))
    visited = set([start_state])
    queue = deque([(blocks, [])])

    while queue:
        cur_blocks, moves = queue.popleft()

        # Ограничение по количеству ходов
        if len(moves) > max_steps:
            return None, None

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



# ------------------------ New generation logic ------------------------


def generate_level_stepwise(settings):
    min_h, max_h = settings["min_h"], settings["max_h"]
    min_v, max_v = settings["min_v"], settings["max_v"]
    min_blockers = settings.get("min_blockers", 0)

    target_h = random.randint(min_h, max_h)
    target_v = random.randint(min_v, max_v)

    letters = list(string.ascii_lowercase)
    random.shuffle(letters)
    letter_i = 0

    key_x = random.choice([0, 1])
    blocks = [Block(KEY_ID, "H", KEY_LEN, key_x, KEY_Y)]

    def try_place_block(block_list, orient):
        nonlocal letter_i
        if letter_i >= len(letters):
            return None

        lid = letters[letter_i]

        # --- Шанс короткого блока ---
        chance_map = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
        chance_short = chance_map.get(min_h if orient == "H" else min_v, 0.1)
        if random.random() < chance_short:
            length = 2
        else:
            length = random.randint(3, BLOCK_MAX)

        occupied = occupied_grid(block_list)
        possible_positions = []

        if orient == "H":
            for y in range(HEIGHT):
                for x in range(WIDTH - length + 1):
                    if all(occupied[y][x + dx] is None for dx in range(length)):
                        possible_positions.append((x, y))
        else:
            for y in range(HEIGHT - length + 1):
                for x in range(WIDTH):
                    if all(occupied[y + dy][x] is None for dy in range(length)):
                        possible_positions.append((x, y))

        random.shuffle(possible_positions)
        if not possible_positions:
            return None

        for x, y in possible_positions:
            new_block = Block(lid, orient, length, x, y)
            test_blocks = block_list + [new_block]
            moves, steps = solve_with_moves(test_blocks)
            if moves is not None:
                letter_i += 1
                return new_block

        return None

    placed_h = 0
    while placed_h < target_h:
        b = try_place_block(blocks, "H")
        if b is None:
            return None
        blocks.append(b)
        placed_h += 1

    placed_v = 0
    while placed_v < target_v:
        b = try_place_block(blocks, "V")
        if b is None:
            return None
        blocks.append(b)
        placed_v += 1

    blockers_count = 0
    key_line_x_end = key_x + KEY_LEN

    occupied = occupied_grid(blocks)
    while blockers_count < min_blockers and letter_i < len(letters):
        lid = letters[letter_i]
        length = random.randint(BLOCK_MIN, BLOCK_MAX)
        y = KEY_Y

        max_x = WIDTH - length
        if max_x < key_line_x_end:
            break

        x_positions = [x for x in range(key_line_x_end, max_x + 1)]
        random.shuffle(x_positions)

        placed = False
        for x in x_positions:
            if all(occupied[y][x + dx] is None for dx in range(length)):
                new_block = Block(lid, "H", length, x, y)
                test_blocks = blocks + [new_block]
                moves, steps = solve_with_moves(test_blocks)
                if moves is not None:
                    blocks.append(new_block)
                    for dx in range(length):
                        occupied[y][x + dx] = lid
                    blockers_count += 1
                    letter_i += 1
                    placed = True
                    break

        if not placed:
            break

    # ------------------------ Лог уровня ------------------------
    # grid = place_blocks(blocks)
    # level_str = grid_to_string(grid)
    # print(f"Level candidate: {level_str}")

    return blocks


# ------------------------ Meta ------------------------


def build_meta_compact(blocks, moves):
    horiz_sizes = [b.length for b in blocks if b.id != KEY_ID and b.orient == "H"]
    vert_sizes = [b.length for b in blocks if b.id != KEY_ID and b.orient == "V"]
    key = next(b for b in blocks if b.id == KEY_ID)
    moves_str = " ".join(f"{bid}{symbol}" for (bid, symbol) in moves)

    grid = place_blocks(blocks)
    empty_cells = sum(row.count("-") for row in grid)
    max_empty_cells = WIDTH * HEIGHT

    min_moves = len(moves)
    max_possible_moves = 40

    difficulty = int(
        min(
            9,
            int(
                min_moves / max_possible_moves * 5
                + (max_empty_cells - empty_cells) / max_empty_cells * 4
            ),
        )
    )

    return {
        "h_blocks": horiz_sizes,
        "v_blocks": vert_sizes,
        "key_x": key.x,
        "min_moves": min_moves,
        "moves": moves_str,
        "difficulty": difficulty,
    }


# ------------------------ JSON writer ------------------------


def _dump_compact_arrays(obj, f, indent=2):
    def _serialize(o, level=0):
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                items.append(
                    " " * (level * indent)
                    + json.dumps(k, ensure_ascii=False)
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


# ------------------------ Master generation ------------------------


def generate_levels(settings, file_path, use_standard_json=False):
    levels_list = []

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, dict) and "levels" in existing:
                    levels_list = existing["levels"]
        except Exception:
            levels_list = []

    for s in settings:
        target = s["count"]
        min_steps_required = s.get("min_steps", 0)

        print(f"Generating {target} levels (min_steps = {min_steps_required})")

        generated = 0
        attempts = 0

        while generated < target:
            attempts += 1

            blocks = generate_level_stepwise(s)
            if blocks is None:
                continue

            moves, steps = solve_with_moves(blocks)
            if moves is None or steps < min_steps_required:
                continue

            level_data = grid_to_string(place_blocks(blocks))
            meta = build_meta_compact(blocks, moves)

            entry = {"data": level_data, "meta": meta}
            levels_list.append(entry)

            save_json(
                {"levels": levels_list},
                file_path,
                indent=2,
                use_standard_json=use_standard_json,
            )

            print(
                f"  ✅ Level {generated + 1} generated successfully "
                f"(min steps: {steps}, difficulty: {meta['difficulty']})"
            )

            generated += 1

        print(f"Finished in {attempts} attempts.")

    return {"levels": levels_list}


# ------------------------ Usage ------------------------


def run():
    file_path = "levels/dataset2.json"
    count = 1000
    settings = [
        {
            "min_h": 1, "max_h": 2,
            "min_v": 1, "max_v": 2,
            "min_blockers": 1,
            "min_steps": 5,
            "count": count,
        },
        {
            "min_h": 2, "max_h": 3,
            "min_v": 2, "max_v": 3,
            "min_blockers": 1,
            "min_steps": 10,
            "count": count,
        },
        {
            "min_h": 2, "max_h": 4,
            "min_v": 2, "max_v": 4,
            "min_blockers": 1,
            "min_steps": 15,
            "count": count,
        },
        {
            "min_h": 3, "max_h": 6,
            "min_v": 3, "max_v": 6,
            "min_blockers": 1,
            "min_steps": 20,
            "count": count,
        },
        {
            "min_h": 5, "max_h": 8,
            "min_v": 5, "max_v": 8,
            "min_blockers": 1,
            "min_steps": 25,
            "count": count,
        },
    ]

    generate_levels(settings, file_path, use_standard_json=False)
    print(f"✅ Finished. Saved to '{file_path}'")


if __name__ == "__main__":
    run()
