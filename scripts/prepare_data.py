#!/usr/bin/env python3
"""Подготовка данных: скачивание датасета и разбиение на клиентов."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.download import download_movielens, load_ratings, load_movies
from src.data.splitter import split_non_iid, split_hybrid_non_iid


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    version = dc.get("version", "1m")  # По умолчанию ML-1M
    use_hybrid = dc.get("use_hybrid", True)  # По умолчанию гибридный сценарий

    print(f"[1/2] Скачиваю MovieLens-{version.upper()}...")
    dp = download_movielens(version=version)
    ratings, item_map = load_ratings(str(dp), version=version)
    movies = load_movies(str(dp), version=version, item_map=item_map)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    print(f"  Юзеров: {n_users}, Фильмов: {n_items}, Рейтингов: {len(ratings)}")

    print(f"\n[2/2] Разбиваю на {dc['num_clients']} клиентов...")

    if use_hybrid:
        print("  Режим: гибридный (публичные + приватные данные)")
        split_hybrid_non_iid(
            ratings, movies,
            num_clients=dc["num_clients"],
            public_user_ratio=dc.get("public_user_ratio", 0.5),
            genre_concentration=dc["genre_concentration"],
            quantity_imbalance=dc["quantity_imbalance"],
            seed=dc["seed"],
        )
    else:
        print("  Режим: чистый non-IID (только приватные данные)")
        split_non_iid(
            ratings, movies,
            num_clients=dc["num_clients"],
            genre_concentration=dc["genre_concentration"],
            quantity_imbalance=dc["quantity_imbalance"],
            seed=dc["seed"],
        )


if __name__ == "__main__":
    main()
