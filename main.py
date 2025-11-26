import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# ===============================
# 1. Настройка API ключей
# ===============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # твой ключ GPT-5
HF_API_TOKEN = os.getenv("HF_API_TOKEN")      # токен Hugging Face

client = OpenAI(api_key=OPENAI_API_KEY)


# ===============================
# 2. Функция генерации уровня GPT-5
# ===============================
def generate_level():
    prompt = """
    Сгенерируй случайный уровень игры 6x6:
    - "-" пустая клетка
    - "0" ключ (2 клетки по горизонтали)
    - "a", "b", "c", ... зеленые свечи (2-5 клетки горизонтально)
    - "x", "y", "z", ... красные свечи (1-5 клеток вертикально)
    Уровень должен быть в формате ASCII grid, 6 строк, 6 колонок, разделитель строк точка '.'
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=150
    )
    level_text = response.choices[0].message.content.strip()
    return level_text


# ===============================
# 3. Функция проверки уровня через Hugging Face RL API
# ===============================
def check_level(level_text):
    """
    Отправляет уровень на RL-агента Hugging Face и получает результат.
    Предполагаем, что модель ожидает JSON вида {"level": "..."}
    """
    url = "https://api-inference.huggingface.co/models/your-username/your-rl-model"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    payload = {"inputs": {"level": level_text}}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("Ошибка запроса к HF RL API:", response.text)
        return None

    result = response.json()
    return result


# ===============================
# 4. Основной pipeline
# ===============================
def main():
    # 1) Генерация уровня
    level = generate_level()
    print("Сгенерированный уровень:")
    print(level)

    # 2) Проверка проходимости через RL-агента
    rl_result = check_level(level)

    if rl_result is None:
        print("Не удалось проверить уровень.")
        return

    # 3) Вывод результатов
    solved = rl_result.get("solved", False)
    steps = rl_result.get("steps", None)
    difficulty = rl_result.get("difficulty", "Unknown")
    solution = rl_result.get("solution", [])

    print("\n=== Результат проверки ===")
    print("Проходимость:", solved)
    print("Количество шагов:", steps)
    print("Сложность:", difficulty)
    print("Пример решения:", solution)


if __name__ == "__main__":
    main()
