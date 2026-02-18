# Data Directory

Датасеты хранятся локально и не коммитятся в git.

## Структура

```
data/
├── raw/                    # исходные датасеты
│   └── ml-100k/           # MovieLens-100K после скачивания
├── processed/              # обработанные данные
│   ├── client_0.pkl       # данные клиента 0
│   ├── client_1.pkl       # данные клиента 1
│   └── ...
└── README.md
```

## Как получить данные

```bash
python scripts/prepare_data.py
```

Скрипт автоматически скачает MovieLens-100K и разобьёт на 20 клиентов.

## Источник

MovieLens-100K: https://grouplens.org/datasets/movielens/100k/
