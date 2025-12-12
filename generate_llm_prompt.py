import json
import random

input_file = "levels/dataset.json"
output_file = "levels/llm_prompt.json"

# Словари для случайной генерации
action_synonyms = ["Сгенерируй", "Создай", "Сделай", "Подготовь"]
block_words = ["блок", "свеча"]
horizontal_words = ["горизонтальный", "зеленый"]
vertical_words = ["вертикальный", "красный"]
size_words = ["небольшим количеством", "большим количеством"]


def generate_block_relation(h_blocks, v_blocks, h_word, v_word, block_word):
    h_count = len(h_blocks)
    v_count = len(v_blocks)

    if h_count > 0 and v_count > 0:
        if h_count == v_count:
            return f"{h_word}ых и {v_word}ых {block_word}ов приблизительно одинаково"
        elif h_count > v_count:
            return f"{h_word}ых {block_word}ов больше, чем {v_word}ых"
        else:
            return f"{h_word}ых {block_word}ов меньше, чем {v_word}ых"
    elif h_count > 0:
        return f"только {h_word}ые {block_word}ы"
    elif v_count > 0:
        return f"только {v_word}ые {block_word}ы"
    else:
        return ""


def generate_human_prompt(level):
    meta = level["meta"]
    moves = meta.get("moves", "")
    h_blocks = meta.get("h_blocks", [])
    v_blocks = meta.get("v_blocks", [])
    key_x = meta.get("key_x", 0)

    # Случайные слова
    action = random.choice(action_synonyms)
    block_word = random.choice(block_words)
    h_word = random.choice(horizontal_words)
    v_word = random.choice(vertical_words)

    # Сложность по ходам
    num_moves = len(moves.split())
    if num_moves <= 10:
        difficulty = "простой"
    elif num_moves < 30:
        difficulty = "средний по сложности"
    else:
        difficulty = "сложный"

    # Количество блоков
    total_blocks = len(h_blocks) + len(v_blocks)
    if total_blocks <= 4:
        blocks_desc = f"с {size_words[0]} {block_word}ов"
    else:
        blocks_desc = f"с {size_words[1]} {block_word}ов"

    # Позиция ключа с градацией
    if key_x <= 1:
        key_pos = "ключ находится далеко от выхода"
    elif key_x <= 3:
        key_pos = "ключ находится на среднем расстоянии от выхода"
    else:
        key_pos = "ключ находится близко к выходу"

    # Цвета/ориентация блоков
    color_desc = generate_block_relation(h_blocks, v_blocks, h_word, v_word, block_word)

    # Пустые клетки
    empty_cells = 36 - sum(h_blocks) - sum(v_blocks)
    empty_desc = "много пустых клеток" if empty_cells >= 18 else "мало пустых клеток"

    # Формируем промпт
    prompt_parts = [difficulty + " уровень на поле 6x6", blocks_desc]
    if color_desc:
        prompt_parts.append(color_desc)
    prompt_parts.append(key_pos)
    prompt_parts.append(empty_desc)

    prompt = action + " " + ", ".join(prompt_parts)
    return prompt


# Загружаем JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Добавляем поле text
for level in data.get("levels", []):
    level["text"] = generate_human_prompt(level)

# Сохраняем новый JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Сохранено в {output_file}")
