import random, json, string, os, time
from collections import namedtuple
from copy import deepcopy

random.seed(20251205)

WIDTH = HEIGHT = 6
KEY_ID = '0'
KEY_Y = 2
KEY_LEN = 2  # ключ всегда длиной 2

MIN_BLOCK = 2   # ✅ минимальный размер блока
MAX_BLOCK = 4   # ✅ максимальный размер блока

Block = namedtuple('Block', ['id', 'orient', 'length', 'x', 'y'])

def occupied_grid(blocks):
    grid = [[None for _ in range(WIDTH)] for __ in range(HEIGHT)]
    for b in blocks:
        if b.orient == 'H':
            for dx in range(b.length):
                grid[b.y][b.x+dx] = b.id
        else:
            for dy in range(b.length):
                grid[b.y+dy][b.x] = b.id
    return grid

def neighbors_single_step(blocks):
    grid = occupied_grid(blocks)
    result = []
    for i,b in enumerate(blocks):
        if b.orient == 'H':
            if b.x - 1 >= 0 and grid[b.y][b.x-1] is None:
                new = list(blocks); new[i] = Block(b.id,b.orient,b.length,b.x-1,b.y)
                result.append(new)
            if b.x + b.length < WIDTH and grid[b.y][b.x + b.length] is None:
                new = list(blocks); new[i] = Block(b.id,b.orient,b.length,b.x+1,b.y)
                result.append(new)
        else:
            if b.y - 1 >= 0 and grid[b.y-1][b.x] is None:
                new = list(blocks); new[i] = Block(b.id,b.orient,b.length,b.x,b.y-1)
                result.append(new)
            if b.y + b.length < HEIGHT and grid[b.y + b.length][b.x] is None:
                new = list(blocks); new[i] = Block(b.id,b.orient,b.length,b.x,b.y+1)
                result.append(new)
    return result

def place_blocks_from_list(blocks):
    grid = [['-' for _ in range(WIDTH)] for __ in range(HEIGHT)]
    for b in blocks:
        if b.orient == 'H':
            for dx in range(b.length):
                grid[b.y][b.x+dx] = b.id
        else:
            for dy in range(b.length):
                grid[b.y+dy][b.x] = b.id
    return grid

def grid_to_level_string(grid):
    return '.'.join(' '.join(row) for row in grid)

def build_initial_blocks(num_blocks_target=10, max_place_attempts=4000):
    letters = list(string.ascii_lowercase)
    blocks = []

    # ✅ Ключ в решённой позиции (у выхода)
    key_block = Block(KEY_ID, 'H', KEY_LEN, WIDTH-KEY_LEN, KEY_Y)
    blocks.append(key_block)

    used = set([(WIDTH-1, KEY_Y), (WIDTH-2, KEY_Y)])
    letter_idx = 0
    attempts = 0

    while len(blocks) < num_blocks_target and attempts < max_place_attempts and letter_idx < len(letters):
        lid = letters[letter_idx]
        letter_idx += 1

        orient = random.choice(['H','V'])
        length = random.randint(MIN_BLOCK, MAX_BLOCK)   # ✅ ТОЛЬКО 2–4

        if orient == 'H':
            y = random.randint(0, HEIGHT-1)
            x = random.randint(0, WIDTH-length)
            if any((x+dx, y) in used for dx in range(length)):
                attempts += 1
                continue
            blocks.append(Block(lid, orient, length, x, y))
            for dx in range(length):
                used.add((x+dx,y))

        else:
            x = random.randint(0, WIDTH-1)
            y = random.randint(0, HEIGHT-length)
            if any((x, y+dy) in used for dy in range(length)):
                attempts += 1
                continue
            blocks.append(Block(lid, orient, length, x, y))
            for dy in range(length):
                used.add((x,y+dy))

    return blocks

def key_x(blocks):
    for b in blocks:
        if b.id == KEY_ID:
            return b.x
    return None

def scramble_to_key_target(blocks, target_x, max_steps=300):
    cur = deepcopy(blocks)
    for _ in range(max_steps):
        if key_x(cur) == target_x:
            break
        nbs = neighbors_single_step(cur)
        if not nbs:
            break
        cur = random.choice(nbs)
    return cur

def extra_scramble_keep_key(cur, moves=20):
    board = deepcopy(cur)
    for _ in range(moves):
        nbs = neighbors_single_step(board)
        if not nbs:
            break
        kx = key_x(board)
        nbs = [nb for nb in nbs if key_x(nb) == kx]
        if not nbs:
            break
        board = random.choice(nbs)
    return board

def generate_level_for_difficulty(num_blocks_range, extra_scramble_range):
    while True:
        nb = random.randint(*num_blocks_range)
        blocks = build_initial_blocks(nb)
        target_x = random.choice([0,1])

        cur = scramble_to_key_target(blocks, target_x)
        if key_x(cur) != target_x:
            continue

        extra = random.randint(*extra_scramble_range)
        cur = extra_scramble_keep_key(cur, extra)

        grid = place_blocks_from_list(cur)
        return grid_to_level_string(grid)

def main():
    result = {"levels": {"simple": [], "medium": [], "difficult": []}}

    for _ in range(2):
        result["levels"]["simple"].append(
            generate_level_for_difficulty((5,7), (3,6))
        )

    for _ in range(2):
        result["levels"]["medium"].append(
            generate_level_for_difficulty((8,10), (6,12))
        )

    for _ in range(1000):
        result["levels"]["difficult"].append(
            generate_level_for_difficulty((11,14), (12,30))
        )

    path = "levels/generated.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ Saved:", path)

if __name__ == "__main__":
    main()
