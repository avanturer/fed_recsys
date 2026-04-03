#!/usr/bin/env python3
"""Подготовка данных: скачивание датасета и разбиение на клиентов."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.download import (
    download_movielens, load_ratings, load_movies,
    download_amazon_music, load_amazon_ratings, load_amazon_items,
)
from src.data.splitter import (
    split_non_iid, split_hybrid_non_iid,
    TARGET_GENRES, TARGET_CATEGORIES_AMAZON,
)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    dataset = dc.get("dataset", f"ml-{dc.get('version', '1m')}")
    use_hybrid = dc.get("use_hybrid", True)

    # Загрузка датасета
    if dataset.startswith("ml-"):
        version = dataset.split("-")[1]
        print(f"[1/2] Скачиваю MovieLens-{version.upper()}...")
        dp = download_movielens(version=version)
        ratings, item_map = load_ratings(str(dp), version=version)
        items = load_movies(str(dp), version=version, item_map=item_map)
        target_genres = TARGET_GENRES
        label = f"MovieLens-{version.upper()}"
    elif dataset == "amazon-music":
        print("[1/2] Скачиваю Amazon Digital Music...")
        dp = download_amazon_music()
        ratings, item_map = load_amazon_ratings(
            str(dp),
            min_user_ratings=dc.get("min_user_ratings", 5),
            min_item_ratings=dc.get("min_item_ratings", 5),
        )
        items = load_amazon_items(str(dp), item_map=item_map)
        target_genres = TARGET_CATEGORIES_AMAZON
        label = "Amazon Digital Music"
    else:
        raise ValueError(f"Неизвестный датасет: {dataset}")

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    print(f"  Датасет: {label}")
    print(f"  Юзеров: {n_users}, Айтемов: {n_items}, Рейтингов: {len(ratings)}")

    print(f"\n[2/2] Разбиваю на {dc['num_clients']} клиентов...")

    if use_hybrid:
        print("  Режим: гибридный (публичные + приватные данные)")
        split_hybrid_non_iid(
            ratings, items,
            num_clients=dc["num_clients"],
            public_user_ratio=dc.get("public_user_ratio", 0.5),
            genre_concentration=dc["genre_concentration"],
            quantity_imbalance=dc["quantity_imbalance"],
            seed=dc["seed"],
            target_genres=target_genres,
        )
    else:
        print("  Режим: чистый non-IID (только приватные данные)")
        split_non_iid(
            ratings, items,
            num_clients=dc["num_clients"],
            genre_concentration=dc["genre_concentration"],
            quantity_imbalance=dc["quantity_imbalance"],
            seed=dc["seed"],
            target_genres=target_genres,
        )


if __name__ == "__main__":
    main()
