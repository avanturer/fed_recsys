"""Скачивание и загрузка MovieLens датасетов (100K и 1M)."""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


def download_movielens(data_dir="data/raw", version="1m"):
    """
    Скачивает MovieLens датасет если ещё не скачан.

    Args:
        data_dir: директория для сырых данных
        version: "100k" или "1m"

    Returns:
        Path к распакованной папке
    """
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    if version == "100k":
        url = ML_100K_URL
        folder_name = "ml-100k"
        zip_name = "ml-100k.zip"
    elif version == "1m":
        url = ML_1M_URL
        folder_name = "ml-1m"
        zip_name = "ml-1m.zip"
    else:
        raise ValueError(f"Неподдерживаемая версия: {version}. Используй '100k' или '1m'")

    extract = path / folder_name
    if extract.exists():
        return extract

    zip_path = path / zip_name
    print(f"Скачиваю MovieLens-{version.upper()}...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)
    zip_path.unlink()

    print(f"Готово: {extract}")
    return extract


def load_ratings(data_dir="data/raw/ml-100k", version="100k"):
    """
    Загружаем рейтинги, переводим ID в contiguous 0-based.

    Args:
        data_dir: путь к папке с данными
        version: "100k" или "1m"

    Returns:
        (DataFrame, item_map) — DataFrame с columns: user_id, item_id, rating, timestamp
        (все ID 0-indexed и contiguous) и маппинг оригинальных item_id → contiguous.
    """
    path = Path(data_dir)

    if version == "100k":
        file_path = path / "u.data"
        df = pd.read_csv(file_path, sep="\t",
                         names=["user_id", "item_id", "rating", "timestamp"],
                         encoding="latin-1")
    elif version == "1m":
        file_path = path / "ratings.dat"
        df = pd.read_csv(file_path, sep="::",
                         names=["user_id", "item_id", "rating", "timestamp"],
                         encoding="latin-1", engine="python")
    else:
        raise ValueError(f"Неподдерживаемая версия: {version}")

    # Перемаппинг в непрерывные 0-indexed ID
    # В ML-1M есть пропуски в ID фильмов (1, 2, 5, 10 -> 0, 1, 2, 3)
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df, item_map


def load_movies(data_dir="data/raw/ml-100k", version="100k", item_map=None):
    """
    Загружаем метаданные фильмов с жанрами.

    Args:
        data_dir: путь к папке с данными
        version: "100k" или "1m"
        item_map: маппинг оригинальных item_id → contiguous (из load_ratings).
                  Необходим для ML-1M, где есть пропуски в ID фильмов.
    """
    path = Path(data_dir)

    if version == "100k":
        file_path = path / "u.item"
        cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRE_COLS
        df = pd.read_csv(file_path, sep="|", names=cols, encoding="latin-1")
    elif version == "1m":
        file_path = path / "movies.dat"
        df = pd.read_csv(file_path, sep="::",
                         names=["item_id", "title", "genres"],
                         encoding="latin-1", engine="python")
        # Конвертируем genres из "Action|Adventure" в one-hot columns
        all_genres = set()
        for g in df["genres"]:
            all_genres.update(g.split("|"))
        all_genres = sorted(all_genres)
        for genre in all_genres:
            df[genre] = df["genres"].str.contains(genre, regex=False).astype(int)
        df = df.drop(columns=["genres"])
    else:
        raise ValueError(f"Неподдерживаемая версия: {version}")

    # Remapping: используем item_map из load_ratings для согласованных ID
    if item_map is not None:
        df["item_id"] = df["item_id"].map(item_map)
        df = df.dropna(subset=["item_id"])
        df["item_id"] = df["item_id"].astype(int)
    else:
        df["item_id"] -= 1

    return df
