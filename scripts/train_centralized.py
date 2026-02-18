#!/usr/bin/env python3
"""
Централизованное обучение (baseline).
Все данные в одном месте — верхняя граница качества,
с которой сравниваем федеративный подход.
"""
import sys
import copy
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.download import download_movielens, load_ratings
from src.models.ncf import NCF
from src.utils.metrics import rmse, mae, evaluate_ranking


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {"user": self.users[idx], "item": self.items[idx],
                "rating": self.ratings[idx]}


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    tc = cfg["training"]
    seed = cfg["data"]["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Данные
    version = cfg["data"].get("version", "1m")
    data_path = download_movielens(version=version)
    ratings, _ = load_ratings(str(data_path), version=version)
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    print(f"Данные: MovieLens-{version.upper()}")
    print(f"        {n_users} юзеров, {n_items} фильмов, {len(ratings)} рейтингов")

    ratings = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(ratings)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)

    train_df = ratings[:n - n_test - n_val]
    val_df = ratings[n - n_test - n_val:n - n_test]
    test_df = ratings[n - n_test:]

    train_loader = DataLoader(RatingsDataset(train_df), batch_size=tc["batch_size"], shuffle=True)
    val_loader = DataLoader(RatingsDataset(val_df), batch_size=tc["batch_size"])
    test_loader = DataLoader(RatingsDataset(test_df), batch_size=tc["batch_size"])

    print(f"Сплит: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # Модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NCF(n_users, n_items, emb_dim=mc["embedding_dim"],
                mlp_layers=mc["mlp_layers"], dropout=mc["dropout"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: {n_params:,} параметров, device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"],
                                  weight_decay=tc.get("weight_decay", 0))
    criterion = torch.nn.MSELoss()

    # Обучение
    n_epochs = tc["centralized_epochs"]
    history = {"train_loss": [], "val_rmse": [], "val_mae": []}
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            u = batch["user"].to(device)
            i = batch["item"].to(device)
            r = batch["rating"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(u, i), r)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # Валидация
        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                u = batch["user"].to(device)
                i = batch["item"].to(device)
                r = batch["rating"].to(device)
                preds_list.extend(model(u, i).cpu().numpy())
                targets_list.extend(r.cpu().numpy())

        p, t = np.array(preds_list), np.array(targets_list)
        v_rmse, v_mae = rmse(p, t), mae(p, t)
        history["val_rmse"].append(v_rmse)
        history["val_mae"].append(v_mae)

        if v_rmse < best_rmse:
            best_rmse = v_rmse
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  val_rmse={v_rmse:.4f}  val_mae={v_mae:.4f}")

    # Тест
    model.load_state_dict(best_state)
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            u = batch["user"].to(device)
            i = batch["item"].to(device)
            r = batch["rating"].to(device)
            preds_list.extend(model(u, i).cpu().numpy())
            targets_list.extend(r.cpu().numpy())

    p, t = np.array(preds_list), np.array(targets_list)
    test_rmse, test_mae = rmse(p, t), mae(p, t)

    # Ранкинговые метрики на тесте
    # Для evaluate_ranking нужны local_user_id / local_item_id — а тут глобальные,
    # поэтому переименовываем
    test_rank = test_df.copy()
    test_rank["local_user_id"] = test_rank["user_id"]
    test_rank["local_item_id"] = test_rank["item_id"]
    all_items = np.arange(n_items)
    ranking = evaluate_ranking(model, test_rank, all_items, ks=(5, 10), device=device)

    print(f"\nТест:  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")
    for k, v in ranking.items():
        print(f"       {k}={v:.4f}")

    # Сохраняем
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, out / "centralized_model.pt")

    history["test_rmse"] = test_rmse
    history["test_mae"] = test_mae
    history["ranking"] = ranking
    with open(out / "centralized_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"\nМодель и история сохранены в {out}")


if __name__ == "__main__":
    main()
