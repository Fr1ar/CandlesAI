# CandlesAI

Установка зависимостей:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запуск тренировки модели:
```bash
python train.py
# Чтобы ноутбук не засыпал:
caffeinate -dims python train.py
```

Запуск решения головоломки:
```bash
python inference.py
```

Форматирование кода:
```bash
ruff format
```
