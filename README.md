# Рекомендательная система на основе гибридного федеративного обучения

ВКР бакалавра, МИСИС, кафедра Инженерной кибернетики.

Тема: *Рекомендательная система на основе гибридного федеративного обучения с разделением параметров модели*

## Архитектура

- **HybridNCF** — NCF с разделением эмбеддингов на публичные (агрегируемые) и приватные (локальные)
- **FedAvg / FedProx** — алгоритмы федеративной агрегации
- **Дифференциальная приватность** — клиппинг обновлений + гауссов шум
- **Flower** — фреймворк для симуляции FL

## Датасеты

- MovieLens-1M (6040 юзеров, 3706 фильмов, 1M рейтингов)
- Amazon Digital Music (рейтинги музыки, k-core фильтрация)

## Запуск

```bash
pip install -r requirements.txt

# 1. Подготовка данных (скачивание + разбиение на клиентов)
python scripts/prepare_data.py

# 2. Централизованный baseline
python scripts/train_centralized.py

# 3. FL-симуляция
python scripts/run_simulation.py

# 4. Графики
python scripts/generate_plots.py

# Полный прогон всех экспериментов (оба датасета, все методы)
python scripts/run_experiments.py
```

## Конфигурация

Все гиперпараметры в `configs/config.yaml`:

```yaml
data:
  dataset: "ml-1m"        # "ml-1m" или "amazon-music"
federated:
  strategy: "fedavg"      # "fedavg" или "fedprox"
  proximal_mu: 0.01       # mu для FedProx
  dp:
    enabled: false         # дифференциальная приватность
    max_grad_norm: 1.0
    noise_multiplier: 0.1
```

## Структура

```
src/
  data/          загрузка и разбиение данных
  models/        NCF, HybridNCF
  federated/     Flower-клиент, сервер, DP
  utils/         метрики (RMSE, MAE, HR@K, NDCG@K)
scripts/         точки входа
configs/         гиперпараметры
report/          LaTeX-отчёт
```

## Стек

Python 3.10, PyTorch, Flower, NumPy, Pandas, Matplotlib, Seaborn
