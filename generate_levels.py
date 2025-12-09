import random, json, string
from collections import deque, namedtuple
import os

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


def neighbors_single_step(blocks):
    grid = occupied_grid(blocks)
    result = []
    for i, b in enumerate(blocks):
        if b.orient == "H":
            if b.x - 1 >= 0 and grid[b.y][b.x - 1] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x - 1, b.y)
                result.append(new)
            if b.x + b.length < WIDTH and grid[b.y][b.x + b.length] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x + 1, b.y)
                result.append(new)
        else:
            if b.y - 1 >= 0 and grid[b.y - 1][b.x] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x, b.y - 1)
                result.append(new)
            if b.y + b.length < HEIGHT and grid[b.y + b.length][b.x] is None:
                new = list(blocks)
                new[i] = Block(b.id, b.orient, b.length, b.x, b.y + 1)
                result.append(new)
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


# ------------------------ BFS Проверка проходимости ------------------------


def is_solvable(blocks):
    start = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in blocks))
    visited = set()
    queue = deque([blocks])
    while queue:
        cur = queue.popleft()
        key = next(b for b in cur if b.id == KEY_ID)
        if key.x + KEY_LEN - 1 == WIDTH - 1 and key.y == KEY_Y:
            return True
        state = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in cur))
        if state in visited:
            continue
        visited.add(state)
        for neigh in neighbors_single_step(cur):
            queue.append(neigh)
    return False


# ------------------------ Подсчёт минимального количества шагов ------------------------


def min_solution_steps(blocks):
    visited = set()
    queue = deque([(blocks, 0)])
    while queue:
        cur, steps = queue.popleft()
        key = next(b for b in cur if b.id == KEY_ID)
        if key.x + KEY_LEN - 1 == WIDTH - 1 and key.y == KEY_Y:
            return steps
        state = tuple(sorted((b.id, b.orient, b.length, b.x, b.y) for b in cur))
        if state in visited:
            continue
        visited.add(state)
        for neigh in neighbors_single_step(cur):
            queue.append((neigh, steps + 1))
    return None


# ------------------------ Массовая генерация с автообновлением файла ------------------------


def generate_levels(settings, file_path):
    result = {s["name"]: [] for s in settings}

    # если файл существует, загружаем промежуточные уровни
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                for k in existing:
                    if k in result:
                        result[k] = existing[k]
            except:
                pass

    for s in settings:
        name = s["name"]
        print(f"Generating {s['count']} levels for '{name}'...")
        generated = len(result[name])
        attempts = 0
        while generated < s["count"]:
            attempts += 1
            blocks = generate_random_blocks_safe(
                s["min_h"], s["max_h"], s["min_v"], s["max_v"], s.get("min_blockers", 0)
            )
            if is_solvable(blocks):
                min_steps_required = s.get("min_steps", 0)
                steps = min_solution_steps(blocks)
                if steps is not None and steps >= min_steps_required:
                    level_str = grid_to_string(place_blocks(blocks))
                    result[name].append(level_str)
                    generated += 1
                    # сохраняем JSON после каждой генерации
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(
                        f"  ✅ Level {generated} for '{name}' generated successfully (min steps: {steps})"
                    )
        print(f"Finished '{name}' in {attempts} attempts.")
    return result


# ------------------------ Пример использования ------------------------

if __name__ == "__main__":
    file_path = "levels/generated_auto.json"
    N = 1000  # количество уровней на категорию
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
