import json

# загрузка уровней из файла
def load_levels(levels_file):
    with open(levels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    levels = data.get("levels", [])
    if not levels:
        raise ValueError("Файл пуст")
    return levels