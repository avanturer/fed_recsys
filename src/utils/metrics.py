"""Метрики для оценки рекомендательной системы."""
import numpy as np
import torch


def rmse(preds, targets):
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def mae(preds, targets):
    return float(np.mean(np.abs(preds - targets)))


def hit_rate_at_k(model, test_pairs, all_item_ids, k=10, n_neg=99, device="cpu", seed=42):
    """
    Hit Rate@K через negative sampling.

    Для каждого (user, true_item) из теста сэмплируем n_neg случайных айтемов,
    считаем скоры модели на все n_neg+1 вариантов, и смотрим попал ли
    true_item в top-K.
    """
    rng = np.random.RandomState(seed)
    model.eval()
    hits = 0

    with torch.no_grad():
        for uid, true_iid in test_pairs:
            pool = all_item_ids[all_item_ids != true_iid]
            neg_items = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
            candidates = np.concatenate([[true_iid], neg_items])

            users_t = torch.full((len(candidates),), uid, dtype=torch.long, device=device)
            items_t = torch.tensor(candidates, dtype=torch.long, device=device)

            scores = model(users_t, items_t).cpu().numpy()

            top_k_idx = np.argsort(scores)[::-1][:k]
            if 0 in top_k_idx:
                hits += 1

    return hits / len(test_pairs) if test_pairs else 0.0


def ndcg_at_k(model, test_pairs, all_item_ids, k=10, n_neg=99, device="cpu", seed=42):
    """
    NDCG@K — то же самое что HR, но с учётом позиции: чем выше true_item
    в ранжировании, тем больше вклад.
    """
    rng = np.random.RandomState(seed)
    model.eval()
    dcg_total = 0.0

    with torch.no_grad():
        for uid, true_iid in test_pairs:
            pool = all_item_ids[all_item_ids != true_iid]
            neg_items = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
            candidates = np.concatenate([[true_iid], neg_items])

            users_t = torch.full((len(candidates),), uid, dtype=torch.long, device=device)
            items_t = torch.tensor(candidates, dtype=torch.long, device=device)

            scores = model(users_t, items_t).cpu().numpy()
            ranked = np.argsort(scores)[::-1]

            pos = np.where(ranked == 0)[0][0]
            if pos < k:
                dcg_total += 1.0 / np.log2(pos + 2)

    idcg = 1.0
    n = len(test_pairs) if test_pairs else 1
    return dcg_total / (n * idcg)


def evaluate_ranking(model, test_df, all_item_ids, ks=(5, 10), device="cpu",
                     n_neg=99, max_pairs=0):
    """
    Считает HR@K и NDCG@K для всех заданных K.
    На вход — test dataframe с local_user_id, local_item_id.
    max_pairs > 0 ограничивает количество тестовых пар (для скорости).
    """
    test_pairs = []
    for uid in test_df["local_user_id"].unique():
        items = test_df[test_df["local_user_id"] == uid]["local_item_id"].values
        if len(items) > 0:
            test_pairs.append((uid, items[-1]))

    if max_pairs > 0 and len(test_pairs) > max_pairs:
        idx = np.random.choice(len(test_pairs), size=max_pairs, replace=False)
        test_pairs = [test_pairs[i] for i in idx]

    results = {}
    for k in ks:
        results[f"hr@{k}"] = hit_rate_at_k(model, test_pairs, all_item_ids,
                                            k=k, n_neg=n_neg, device=device)
        results[f"ndcg@{k}"] = ndcg_at_k(model, test_pairs, all_item_ids,
                                          k=k, n_neg=n_neg, device=device)
    return results
