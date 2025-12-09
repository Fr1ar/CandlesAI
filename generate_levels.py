import random, json, string, os
from collections import namedtuple
from copy import deepcopy

WIDTH = HEIGHT = 6
KEY_ID = "0"
KEY_Y = 2
KEY_LEN = 2

MIN_BLOCK = 2
MAX_BLOCK = 4

Block = namedtuple("Block", ["id", "orient", "length", "x", "y"])

# ------------------- БАЗОВАЯ МЕХАНИКА -------------------


def occupied_grid(blocks):
    grid = [[None for _ in range(WIDTH)] for __ in range(HEIGHT)]
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


# ------------------- ПОИСК ЗАТОРОВ -------------------


def count_real_blockers(blocks):
    """
    Считает сколько РЕАЛЬНЫХ блоков физически перекрывают путь ключа к выходу.
    """
    grid = occupied_grid(blocks)
    key = next(b for b in blocks if b.id == KEY_ID)
    blockers = set()

    y = key.y
    for x in range(key.x + key.length, WIDTH):
        cell = grid[y][x]
        if cell and cell != KEY_ID:
            blockers.add(cell)

    return len(blockers)


# ------------------- ГЕНЕРАЦИЯ РЕШЁННОЙ ДОСКИ -------------------


def build_solved_board(num_blocks):
    letters = list(string.ascii_lowercase)
    blocks = []

    # Ключ у выхода
    blocks.append(Block(KEY_ID, "H", KEY_LEN, WIDTH - KEY_LEN, KEY_Y))
    used = {(WIDTH - 1, KEY_Y), (WIDTH - 2, KEY_Y)}

    letter_idx = 0
    while len(blocks) < num_blocks and letter_idx < len(letters):
        lid = letters[letter_idx]
        letter_idx += 1

        orient = random.choice(["H", "V"])
        length = random.randint(MIN_BLOCK, MAX_BLOCK)

        if orient == "H":
            y = random.randint(0, HEIGHT - 1)
            x = random.randint(0, WIDTH - length)
            if any((x + dx, y) in used for dx in range(length)):
                continue
            blocks.append(Block(lid, orient, length, x, y))
            for dx in range(length):
                used.add((x + dx, y))
        else:
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - length)
            if any((x, y + dy) in used for dy in range(length)):
                continue
            blocks.append(Block(lid, orient, length, x, y))
            for dy in range(length):
                used.add((x, y + dy))

    return blocks


# ------------------- СКРЭМБЛИНГ -------------------


def key_x(blocks):
    return next(b.x for b in blocks if b.id == KEY_ID)


def scramble_to_start(blocks):
    target_x = random.choice([0, 1])
    cur = deepcopy(blocks)

    for _ in range(300):
        if key_x(cur) == target_x:
            return cur
        nbs = neighbors_single_step(cur)
        if not nbs:
            break
        cur = random.choice(nbs)

    return None


def extra_scramble_keep_key(board, steps):
    cur = deepcopy(board)
    for _ in range(steps):
        nbs = neighbors_single_step(cur)
        kx = key_x(cur)
        nbs = [b for b in nbs if key_x(b) == kx]
        if not nbs:
            break
        cur = random.choice(nbs)
    return cur


# ------------------- ГЕНЕРАЦИЯ УРОВНЕЙ С ЗАТОРАМИ -------------------


def generate_level(block_range, scramble_range, min_blockers):
    while True:
        blocks = build_solved_board(random.randint(*block_range))
        scrambled = scramble_to_start(blocks)
        if not scrambled:
            continue

        scrambled = extra_scramble_keep_key(scrambled, random.randint(*scramble_range))
        blockers = count_real_blockers(scrambled)

        if blockers >= min_blockers:
            return grid_to_string(place_blocks(scrambled))


# ------------------- ГЛАВНАЯ ФУНКЦИЯ -------------------


def main():
    result = {"levels": []}

    # ✅ СЛОЖНЫЙ — минимум 3 затора
    for i in range(1000):
        result["levels"].append(generate_level((11, 14), (16, 30), min_blockers=3))
        if i % 10 == 0:
            print(f"Уровней сгенерировано: {i}")

    with open("levels/generated.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ Файл сохранён")


if __name__ == "__main__":
    main()
