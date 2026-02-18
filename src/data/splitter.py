"""Разбиение датасета на клиентов с non-IID распределением."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# Жанры для распределения по клиентам
TARGET_GENRES = [
    "Action", "Comedy", "Drama", "Romance", "Thriller",
    "Sci-Fi", "Horror", "Adventure", "Crime", "Fantasy"
]


class ClientDataset(Dataset):
    """Датасет одного клиента для PyTorch DataLoader."""

    def __init__(self, df):
        self.users = df["local_user_id"].values
        self.items = df["local_item_id"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
        }


class HybridClientDataset(Dataset):
    """Датасет для гибридного сценария (публичные + приватные юзеры)."""

    def __init__(self, df):
        """
        df должен содержать:
        - is_public: 1 если публичный юзер, 0 если приватный
        - public_user_id: ID для публичных (NaN для приватных)
        - private_user_id: ID для приватных (NaN для публичных)
        - item_id: глобальный ID айтема
        - rating: рейтинг
        """
        self.is_public = df["is_public"].values.astype(bool)

        # Для публичных юзеров берём public_user_id, для приватных — private_user_id
        self.users = np.where(
            self.is_public,
            df["public_user_id"].fillna(0).values,
            df["private_user_id"].fillna(0).values
        ).astype(np.int64)

        self.items = df["item_id"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user": self.users[idx],
            "item": self.items[idx],
            "rating": self.ratings[idx],
            "is_public": self.is_public[idx],
        }


def _assign_primary_genre(movies):
    """Для каждого фильма определяем основной жанр из TARGET_GENRES."""
    genre_cols = [g for g in TARGET_GENRES if g in movies.columns]
    mapping = {}
    for _, row in movies.iterrows():
        genres = [g for g in genre_cols if row[g] == 1]
        mapping[row["item_id"]] = genres[0] if genres else "Other"
    return mapping, genre_cols


def _find_user_genre_pref(ratings):
    """Для каждого юзера считаем какой жанр он смотрит чаще всего."""
    prefs = {}
    for uid in ratings["user_id"].unique():
        counts = ratings[ratings["user_id"] == uid]["genre"].value_counts()
        prefs[uid] = counts.index[0] if len(counts) > 0 else "Other"
    return prefs


def split_non_iid(ratings, movies, num_clients=20,
                  genre_concentration=0.6, quantity_imbalance=0.5,
                  seed=42, output_dir="data/processed"):
    """
    Разбивает рейтинги на клиентов с non-IID распределением.

    Каждому клиенту назначается «основной» и «побочный» жанр.
    Юзеры распределяются по клиентам с учётом жанровых предпочтений,
    плюс дирихле-шум для имитации разного объёма данных.
    Один юзер — один клиент (данные не пересекаются).
    """
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    item_genre_map, genre_cols = _assign_primary_genre(movies)

    ratings = ratings.copy()
    ratings["genre"] = ratings["item_id"].map(item_genre_map).fillna("Other")

    # Каждому клиенту — свои жанровые предпочтения
    client_genres = []
    for i in range(num_clients):
        main = genre_cols[i % len(genre_cols)]
        secondary = genre_cols[(i + 3) % len(genre_cols)]
        conc = genre_concentration + np.random.uniform(-0.1, 0.1)
        client_genres.append({"main": main, "secondary": secondary,
                              "concentration": conc})

    # Дирихле для имитации неравномерного количества данных
    alpha = max(0.3, 1.0 - quantity_imbalance * 0.5)
    qty_weights = np.random.dirichlet([alpha] * num_clients)

    all_users = ratings["user_id"].unique()
    np.random.shuffle(all_users)
    user_pref = _find_user_genre_pref(ratings)

    # Сначала гарантируем каждому клиенту минимум юзеров
    min_users = max(5, len(all_users) // (num_clients * 3))
    buckets = [[] for _ in range(num_clients)]
    idx = 0
    for cid in range(num_clients):
        for _ in range(min_users):
            if idx < len(all_users):
                buckets[cid].append(all_users[idx])
                idx += 1

    # Оставшихся раскидываем с учётом жанровых предпочтений
    for uid in all_users[idx:]:
        pref = user_pref.get(uid, "Other")

        matching = [c for c, cg in enumerate(client_genres)
                    if cg["main"] == pref or cg["secondary"] == pref]

        if matching and np.random.random() < genre_concentration:
            w = np.array([qty_weights[c] for c in matching])
            w /= w.sum()
            chosen = np.random.choice(matching, p=w)
        else:
            chosen = np.random.choice(num_clients, p=qty_weights)

        buckets[chosen].append(uid)

    # Формируем датасеты клиентов
    clients = []
    print(f"\n{'='*65}")
    print(f"Non-IID разбиение на {num_clients} клиентов")
    print(f"{'='*65}")

    for cid in range(num_clients):
        users = buckets[cid]
        if not users:
            continue

        cr = ratings[ratings["user_id"].isin(users)].copy()

        local_u = {u: i for i, u in enumerate(cr["user_id"].unique())}
        local_i = {it: i for i, it in enumerate(cr["item_id"].unique())}
        cr["local_user_id"] = cr["user_id"].map(local_u)
        cr["local_item_id"] = cr["item_id"].map(local_i)

        train, tmp = train_test_split(cr, test_size=0.2, random_state=seed)
        val, test = train_test_split(tmp, test_size=0.5, random_state=seed)

        genre_dist = cr["genre"].value_counts(normalize=True)
        main_g = client_genres[cid]["main"]

        info = {
            "client_id": cid,
            "num_users": len(local_u),
            "num_items": len(local_i),
            "train": train, "val": val, "test": test,
            "user_mapping": local_u,
            "item_mapping": local_i,
            "main_genre": main_g,
            "genre_distribution": genre_dist.to_dict(),
        }
        clients.append(info)

        with open(out / f"client_{cid}.pkl", "wb") as f:
            pickle.dump(info, f)

        top3 = ", ".join(f"{g}:{p*100:.0f}%" for g, p in list(genre_dist.items())[:3])
        pct = genre_dist.get(main_g, 0) * 100
        print(f"  Клиент {cid:2d}: {len(users):3d} юзеров, "
              f"{len(train):4d} train | {main_g:10s} ({pct:4.1f}%) | {top3}")

    # Сохраняем общую информацию
    global_info = {
        "num_clients": len(clients),
        "total_users": ratings["user_id"].nunique(),
        "total_items": ratings["item_id"].nunique(),
        "total_ratings": len(ratings),
        "genre_concentration": genre_concentration,
        "quantity_imbalance": quantity_imbalance,
        "client_genres": client_genres,
    }
    with open(out / "global_info.pkl", "wb") as f:
        pickle.dump(global_info, f)

    sizes = [len(c["train"]) for c in clients]
    print(f"\n  Разброс: min={min(sizes)}, max={max(sizes)}, "
          f"ratio={max(sizes)/max(min(sizes),1):.1f}x")
    print(f"  Сохранено {len(clients)} клиентов в {out}")
    print(f"{'='*65}")

    return clients


def split_hybrid_non_iid(ratings, movies, num_clients=20,
                          public_user_ratio=0.5, genre_concentration=0.6,
                          quantity_imbalance=0.5, seed=42,
                          output_dir="data/processed"):
    """
    Гибридный сценарий: публичные + приватные данные.

    50% пользователей — «публичные» (все клиенты видят их рейтинги).
    50% пользователей — «приватные» (распределены non-IID по клиентам).

    Каждый клиент получает:
    - Все рейтинги публичных юзеров
    - Рейтинги своих приватных юзеров

    Public user IDs — глобальные (одинаковые на всех клиентах).
    Private user IDs — локальные (0-indexed отдельно для каждого клиента).
    Item IDs — глобальные (одинаковые на всех клиентах).

    Args:
        public_user_ratio: доля публичных пользователей (0.0-1.0)
    """
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    item_genre_map, genre_cols = _assign_primary_genre(movies)
    ratings = ratings.copy()
    ratings["genre"] = ratings["item_id"].map(item_genre_map).fillna("Other")

    all_users = ratings["user_id"].unique()
    np.random.shuffle(all_users)

    # Разделяем юзеров на публичных и приватных
    n_public = int(len(all_users) * public_user_ratio)
    public_users = all_users[:n_public]
    private_users = all_users[n_public:]

    print(f"\n{'='*65}")
    print(f"Гибридное разбиение: {n_public} публичных + {len(private_users)} приватных юзеров")
    print(f"{'='*65}")

    public_ratings = ratings[ratings["user_id"].isin(public_users)].copy()
    private_ratings = ratings[ratings["user_id"].isin(private_users)].copy()

    # Публичные юзеры: global user IDs (0-indexed)
    public_user_map = {u: i for i, u in enumerate(sorted(public_users))}
    public_ratings["public_user_id"] = public_ratings["user_id"].map(public_user_map)

    # Айтемы: глобальные ID (уже 0-indexed после load_ratings)
    all_items = ratings["item_id"].unique()
    item_map = {it: it for it in all_items}

    # Распределяем приватных юзеров по клиентам (non-IID)
    user_pref = _find_user_genre_pref(private_ratings)

    # Каждому клиенту — свои жанровые предпочтения
    client_genres = []
    for i in range(num_clients):
        main = genre_cols[i % len(genre_cols)]
        secondary = genre_cols[(i + 3) % len(genre_cols)]
        conc = genre_concentration + np.random.uniform(-0.1, 0.1)
        client_genres.append({"main": main, "secondary": secondary,
                              "concentration": conc})

    # Дирихле для имитации неравномерного количества данных
    alpha = max(0.3, 1.0 - quantity_imbalance * 0.5)
    qty_weights = np.random.dirichlet([alpha] * num_clients)

    # Распределяем приватных юзеров
    min_users = max(5, len(private_users) // (num_clients * 3))
    buckets = [[] for _ in range(num_clients)]
    idx = 0

    for cid in range(num_clients):
        for _ in range(min_users):
            if idx < len(private_users):
                buckets[cid].append(private_users[idx])
                idx += 1

    for uid in private_users[idx:]:
        pref = user_pref.get(uid, "Other")
        matching = [c for c, cg in enumerate(client_genres)
                    if cg["main"] == pref or cg["secondary"] == pref]

        if matching and np.random.random() < genre_concentration:
            w = np.array([qty_weights[c] for c in matching])
            w /= w.sum()
            chosen = np.random.choice(matching, p=w)
        else:
            chosen = np.random.choice(num_clients, p=qty_weights)

        buckets[chosen].append(uid)

    # Формируем датасеты клиентов
    clients = []

    for cid in range(num_clients):
        private_users_cid = buckets[cid]
        if not private_users_cid:
            private_users_cid = []

        # Приватные данные клиента
        private_cr = private_ratings[private_ratings["user_id"].isin(private_users_cid)].copy()
        local_private_user_map = {u: i for i, u in enumerate(sorted(private_users_cid))}
        private_cr["private_user_id"] = private_cr["user_id"].map(local_private_user_map)

        # Объединяем публичные + приватные данные
        # Публичные: (public_user_id, item_id, rating)
        # Приватные: (private_user_id, item_id, rating)
        public_cr = public_ratings.copy()
        public_cr["is_public"] = 1
        private_cr["is_public"] = 0

        # Все данные клиента
        client_data = pd.concat([public_cr, private_cr], ignore_index=True)

        # Разбиваем на train/val/test
        train, tmp = train_test_split(client_data, test_size=0.2, random_state=seed)
        val, test = train_test_split(tmp, test_size=0.5, random_state=seed)

        genre_dist_all = client_data["genre"].value_counts(normalize=True)
        main_g = client_genres[cid]["main"]

        info = {
            "client_id": cid,
            "num_public_users": len(public_user_map),
            "num_private_users": len(local_private_user_map),
            "num_items": len(all_items),
            "train": train, "val": val, "test": test,
            "public_user_mapping": public_user_map,
            "private_user_mapping": local_private_user_map,
            "item_mapping": item_map,
            "main_genre": main_g,
            "genre_distribution": genre_dist_all.to_dict(),
        }
        clients.append(info)

        with open(out / f"client_{cid}.pkl", "wb") as f:
            pickle.dump(info, f)

        top3 = ", ".join(f"{g}:{p*100:.0f}%" for g, p in list(genre_dist_all.items())[:3])
        pct = genre_dist_all.get(main_g, 0) * 100
        n_pub_ratings = len(train[train["is_public"] == 1])
        n_priv_ratings = len(train[train["is_public"] == 0])
        print(f"  Клиент {cid:2d}: pub={n_pub_ratings:4d} priv={n_priv_ratings:4d} | "
              f"{main_g:10s} ({pct:4.1f}%) | {top3}")

    # Сохраняем общую информацию
    global_info = {
        "num_clients": len(clients),
        "num_public_users": len(public_user_map),
        "total_users": len(all_users),
        "total_items": len(all_items),
        "total_ratings": len(ratings),
        "public_user_ratio": public_user_ratio,
        "genre_concentration": genre_concentration,
        "quantity_imbalance": quantity_imbalance,
        "client_genres": client_genres,
        "is_hybrid": True,
    }
    with open(out / "global_info.pkl", "wb") as f:
        pickle.dump(global_info, f)

    sizes = [len(c["train"]) for c in clients]
    print(f"\n  Разброс train: min={min(sizes)}, max={max(sizes)}, "
          f"ratio={max(sizes)/max(min(sizes),1):.1f}x")
    print(f"  Публичных юзеров: {len(public_user_map)}")
    print(f"  Сохранено {len(clients)} клиентов в {out}")
    print(f"{'='*65}")

    return clients


def load_client_data(client_id, data_dir="data/processed"):
    with open(Path(data_dir) / f"client_{client_id}.pkl", "rb") as f:
        return pickle.load(f)


def load_global_info(data_dir="data/processed"):
    with open(Path(data_dir) / "global_info.pkl", "rb") as f:
        return pickle.load(f)
