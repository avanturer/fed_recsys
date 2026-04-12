# Data Directory

Датасеты хранятся локально и не коммитятся в git.

## Структура

```
data/
├── raw/                    # исходные датасеты
│   ├── ml-1m/             # MovieLens-1M
│   └── amazon-music/      # Amazon Digital Music
├── processed/              # обработанные данные клиентов
│   ├── global_info.pkl    # общая информация о разбиении
│   ├── client_0.pkl       # данные клиента 0
│   ├── client_1.pkl       # данные клиента 1
│   └── ...
├── results/                # результаты экспериментов
└── README.md
```

## Как получить данные

```bash
python scripts/prepare_data.py
```

Датасет задаётся в `configs/config.yaml` (поле `data.dataset`).
Скрипт автоматически скачает данные и разобьёт на клиентов.

## Источники

- MovieLens-1M: https://grouplens.org/datasets/movielens/1m/
- Amazon Digital Music: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
