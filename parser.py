import json

# Загрузка уровней из файла
def load_levels(levels_file):
    with open(levels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    levels = data.get("levels", [])
    if not levels:
        raise ValueError("Файл пуст")
    return levels


# Парсер уровня
def parse_level(text_level):
    next_id = 0
    blocks = {}
    block_texts = {}
    key_id = None

    lines = text_level.split(".")
    grid = [row.split() for row in lines]

    H, W = len(grid), len(grid[0])
    used = [[False] * W for _ in range(H)]

    for y in range(H):
        for x in range(W):
            if grid[y][x] == "-" or used[y][x]:
                continue

            ch = grid[y][x]

            # horizontal?
            w = 1
            while x + w < W and grid[y][x + w] == ch:
                w += 1

            if w > 1:
                blocks[next_id] = {"x": x, "y": y, "w": w, "h": 1, "type": "H"}
                block_texts[next_id] = ch
                if ch == "0":
                    key_id = next_id

                for dx in range(w):
                    used[y][x + dx] = True

                next_id += 1
                continue

            # vertical?
            h = 1
            while y + h < H and grid[y + h][x] == ch:
                h += 1

            if h > 1:
                blocks[next_id] = {"x": x, "y": y, "w": 1, "h": h, "type": "V"}
                block_texts[next_id] = ch
                if ch == "0":
                    key_id = next_id

                for dy in range(h):
                    used[y + dy][x] = True

                next_id += 1
                continue

            # single block
            blocks[next_id] = {
                "x": x, "y": y, "w": 1, "h": 1,
                "type": "H" if ch == "0" else "V"
            }
            block_texts[next_id] = ch
            if ch == "0":
                key_id = next_id

            used[y][x] = True
            next_id += 1

    if key_id is None:
        raise ValueError("No key '0' found")

    return blocks, block_texts, key_id


# Генерация уровня по умолчанию
def generate_default_level():
    next_id = 0
    blocks = {}
    block_texts = {}

    blocks[next_id] = {"x": 0, "y": 2, "w": 2, "h": 1, "type": "H"}
    block_texts[next_id] = "0"
    key_id = next_id
    next_id += 1

    blocks[next_id] = {"x": 2, "y": 0, "w": 3, "h": 1, "type": "H"}
    block_texts[next_id] = "a"
    next_id += 1

    blocks[next_id] = {"x": 3, "y": 1, "w": 1, "h": 3, "type": "V"}
    block_texts[next_id] = "b"
    next_id += 1

    return blocks, block_texts, key_id