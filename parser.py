import json


# ----------------- Загрузка уровней из файла -----------------
def load_levels(levels_file):
    with open(levels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    levels_raw = data.get("levels", [])
    if not levels_raw:
        raise ValueError("Файл пуст")

    levels = []
    for i, lvl in enumerate(levels_raw):
        text = lvl.get("data", "")
        if not text:
            raise ValueError(f"Уровень {i} не содержит поле 'data'")
        meta = lvl.get("meta", None)
        levels.append({"data": text, "meta": meta})
    return levels


# ----------------- Парсер уровня -----------------
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

            # горизонтальный блок
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

            # вертикальный блок
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

            # одиночный блок
            blocks[next_id] = {
                "x": x,
                "y": y,
                "w": 1,
                "h": 1,
                "type": "H" if ch == "0" else "V",
            }
            block_texts[next_id] = ch
            if ch == "0":
                key_id = next_id

            used[y][x] = True
            next_id += 1

    if key_id is None:
        raise ValueError("No key '0' found")

    return blocks, block_texts, key_id


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
