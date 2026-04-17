"""Скачивание и загрузка датасетов (MovieLens, Amazon Digital Music)."""
import gzip
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

AMAZON_MUSIC_REVIEWS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/Digital_Music.jsonl"
)
AMAZON_MUSIC_META_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_Digital_Music.jsonl"
)

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


# --- Amazon Digital Music ---

MUSIC_CATEGORIES = [
    "Rock", "Pop", "Jazz", "Classical", "Country",
    "R&B", "Hip-Hop", "Electronic", "Folk", "Blues",
    "Metal", "Soul", "Reggae", "Latin", "Alternative",
]


def download_amazon_music(data_dir="data/raw"):
    """Скачивает Amazon Digital Music (reviews + metadata)."""
    path = Path(data_dir) / "amazon-music"
    path.mkdir(parents=True, exist_ok=True)

    reviews = path / "Digital_Music.jsonl"
    meta = path / "meta_Digital_Music.jsonl"

    # Совместимость: если есть старый .gz — тоже подходит
    reviews_gz = path / "Digital_Music.jsonl.gz"
    meta_gz = path / "meta_Digital_Music.jsonl.gz"

    if not reviews.exists() and not reviews_gz.exists():
        print("Скачиваю Amazon Digital Music (reviews)...")
        urllib.request.urlretrieve(AMAZON_MUSIC_REVIEWS_URL, reviews)
        print(f"  Готово: {reviews}")

    if not meta.exists() and not meta_gz.exists():
        print("Скачиваю Amazon Digital Music (metadata)...")
        urllib.request.urlretrieve(AMAZON_MUSIC_META_URL, meta)
        print(f"  Готово: {meta}")

    return path


def _k_core_filter(df, min_user=5, min_item=5, max_iter=100):
    """Итеративная k-core фильтрация: убираем юзеров/айтемов с < k оценок."""
    for _ in range(max_iter):
        prev = len(df)
        if df.empty:
            break
        uc = df["user_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        ic = df["item_id"].value_counts()
        df = df[df["item_id"].isin(ic[ic >= min_item].index)]
        if len(df) == prev:
            break
    if df.empty:
        raise ValueError(
            f"k-core фильтрация ({min_user}/{min_item}) удалила все данные. "
            f"Попробуйте уменьшить min_user_ratings/min_item_ratings."
        )
    return df


def load_amazon_ratings(data_dir="data/raw/amazon-music",
                        min_user_ratings=5, min_item_ratings=5):
    """
    Загружаем рейтинги Amazon Digital Music, k-core фильтрация,
    ремаппинг в contiguous 0-indexed ID.

    Returns:
        (DataFrame[user_id, item_id, rating, timestamp], item_map)
    """
    path = Path(data_dir)
    reviews_gz = path / "Digital_Music.jsonl.gz"
    reviews_plain = path / "Digital_Music.jsonl"
    reviews_file = reviews_gz if reviews_gz.exists() else reviews_plain

    rows = []
    opener = gzip.open if reviews_file.suffix == ".gz" else open
    with opener(reviews_file, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Совместимость с разными версиями Amazon Review Dataset (2018/2023)
            uid = rec.get("user_id", rec.get("reviewerID", ""))
            iid = rec.get("parent_asin", rec.get("asin", ""))
            score = float(rec.get("rating", rec.get("overall", 0)))
            ts = int(rec.get("timestamp", rec.get("unixReviewTime", 0)))
            if uid and iid and score > 0:
                rows.append({
                    "user_id": uid,
                    "item_id": iid,
                    "rating": score,
                    "timestamp": ts,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Не удалось загрузить рейтинги из {reviews_file}")
    print(f"  Сырых рейтингов: {len(df)}, "
          f"юзеров: {df['user_id'].nunique()}, "
          f"айтемов: {df['item_id'].nunique()}")

    # k-core фильтрация
    df = _k_core_filter(df, min_user=min_user_ratings, min_item=min_item_ratings)
    print(f"  После {min_user_ratings}-core: {len(df)} рейтингов, "
          f"юзеров: {df['user_id'].nunique()}, "
          f"айтемов: {df['item_id'].nunique()}")

    # Ремаппинг в contiguous 0-indexed
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {old: new for new, old in enumerate(unique_users)}
    item_map = {old: new for new, old in enumerate(unique_items)}

    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    return df, item_map


def load_amazon_items(data_dir="data/raw/amazon-music", item_map=None):
    """
    Загружаем метаданные Amazon Digital Music, извлекаем категории.

    Returns:
        DataFrame с columns [item_id, title, Rock, Pop, Jazz, ...]
    """
    path = Path(data_dir)
    meta_gz = path / "meta_Digital_Music.jsonl.gz"
    meta_plain = path / "meta_Digital_Music.jsonl"
    meta_file = meta_gz if meta_gz.exists() else meta_plain

    items = []
    opener = gzip.open if meta_file.suffix == ".gz" else open
    with opener(meta_file, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            asin = rec.get("parent_asin", rec.get("asin", ""))
            title = rec.get("title", "")
            # Разные версии используют categories / category / main_category
            cats = rec.get("categories", rec.get("category", []))
            if isinstance(cats, str):
                cats = [cats]
            if cats and isinstance(cats[0], list):
                cats = [c for sub in cats for c in sub]
            # main_category — часто самая информативная
            main_cat = rec.get("main_category", "")
            if main_cat and main_cat not in cats:
                cats.append(main_cat)
            items.append({"item_id": asin, "title": title, "categories": cats})

    df = pd.DataFrame(items)

    # One-hot по целевым категориям
    for cat in MUSIC_CATEGORIES:
        df[cat] = df["categories"].apply(
            lambda cs: int(any(cat.lower() in c.lower() for c in cs))
        )
    df = df.drop(columns=["categories"])

    # Ремаппинг
    if item_map is not None:
        df["item_id"] = df["item_id"].map(item_map)
        df = df.dropna(subset=["item_id"])
        df["item_id"] = df["item_id"].astype(int)

    return df
