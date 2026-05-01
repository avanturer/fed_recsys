"""
Серверная часть: стратегия FedAvg и запуск симуляции.

Сервер агрегирует только shared-параметры (MLP + output)
от клиентов. Эмбеддинги живут локально — сервер их не видит.
После симуляции — оценка финальной модели на тестовых данных клиентов.
"""
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import flwr as fl

from src.models.ncf import NCF, HybridNCF
from src.data.splitter import load_global_info, load_client_data
from src.federated.client import make_client_fn, make_hybrid_client_fn
from src.federated.privacy import PrivacyAccountant
from src.utils.metrics import rmse, mae, evaluate_ranking, hit_rate_at_k, ndcg_at_k


def _weighted_avg(metrics_list):
    """Взвешенное среднее метрик от клиентов (по количеству примеров)."""
    total = sum(n for n, _ in metrics_list)
    if total == 0:
        return {}
    result = {}
    keys = metrics_list[0][1].keys()
    for k in keys:
        result[k] = sum(n * m[k] for n, m in metrics_list) / total
    return result


class _SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg, которая сохраняет агрегированные параметры после каждого раунда."""

    def __init__(self, save_path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params = fl.common.parameters_to_ndarrays(aggregated)
            with open(self.save_path, "wb") as f:
                pickle.dump(params, f)
        return aggregated, metrics


def _post_fl_evaluation(shared_params, cfg, data_dir):
    """
    Оценка финальной FL-модели: для каждого клиента собираем модель
    (shared MLP с сервера + локальные эмбеддинги с диска) и считаем метрики на тесте.
    """
    info = load_global_info(data_dir)
    emb_cache = Path(data_dir) / "embed_cache"

    all_rmse, all_mae = [], []
    all_hr5, all_ndcg5 = [], []
    all_hr10, all_ndcg10 = [], []

    print(f"\n{'='*55}")
    print("Оценка финальной FL-модели на тестовых данных")
    print(f"{'='*55}")

    for cid in range(info["num_clients"]):
        data = load_client_data(cid, data_dir)

        model = NCF(
            data["num_users"], data["num_items"],
            emb_dim=cfg["emb_dim"],
            mlp_layers=cfg.get("mlp_layers"),
            dropout=cfg.get("dropout", 0.2),
        )

        # Shared-параметры с сервера
        model.load_shared_params(shared_params)

        # Локальные эмбеддинги с диска
        emb_path = emb_cache / f"client_{cid}_emb.pt"
        if emb_path.exists():
            emb_state = torch.load(emb_path, map_location="cpu", weights_only=True)
            state = model.state_dict()
            for k, v in emb_state.items():
                if k in state and state[k].shape == v.shape:
                    state[k] = v
            model.load_state_dict(state)

        model.eval()
        test_df = data["test"]

        with torch.no_grad():
            users_t = torch.tensor(test_df["local_user_id"].values, dtype=torch.long)
            items_t = torch.tensor(test_df["local_item_id"].values, dtype=torch.long)
            ratings = test_df["rating"].values.astype(np.float32)
            preds = model(users_t, items_t).numpy()

        c_rmse = rmse(preds, ratings)
        c_mae = mae(preds, ratings)
        all_rmse.append(c_rmse)
        all_mae.append(c_mae)

        # Ранкинговые метрики (n_neg=99 — такой же протокол что и у централизованной)
        all_items = np.arange(data["num_items"])
        ranking = evaluate_ranking(
            model, test_df, all_items,
            ks=(5, 10), device="cpu", n_neg=99,
        )
        all_hr5.append(ranking["hr@5"])
        all_ndcg5.append(ranking["ndcg@5"])
        all_hr10.append(ranking["hr@10"])
        all_ndcg10.append(ranking["ndcg@10"])

        print(f"  Клиент {cid:2d}: RMSE={c_rmse:.4f}  HR@10={ranking['hr@10']:.3f}")

    results = {
        "test_rmse": float(np.mean(all_rmse)),
        "test_mae": float(np.mean(all_mae)),
        "test_hr5": float(np.mean(all_hr5)),
        "test_ndcg5": float(np.mean(all_ndcg5)),
        "test_hr10": float(np.mean(all_hr10)),
        "test_ndcg10": float(np.mean(all_ndcg10)),
        "per_client_rmse": all_rmse,
        "per_client_hr10": all_hr10,
    }

    print(f"\n  Среднее по клиентам:")
    print(f"    RMSE    = {results['test_rmse']:.4f}")
    print(f"    MAE     = {results['test_mae']:.4f}")
    print(f"    HR@5    = {results['test_hr5']:.4f}")
    print(f"    NDCG@5  = {results['test_ndcg5']:.4f}")
    print(f"    HR@10   = {results['test_hr10']:.4f}")
    print(f"    NDCG@10 = {results['test_ndcg10']:.4f}")

    return results


def _build_hybrid_model(data, shared_params, emb_cache, cid, cfg):
    """Собирает HybridNCF клиента: shared params с сервера + его приватные эмбеддинги."""
    model = HybridNCF(
        num_public_users=data["num_public_users"],
        num_private_users=data["num_private_users"],
        num_items=data["num_items"],
        emb_dim=cfg["emb_dim"],
        mlp_layers=cfg.get("mlp_layers"),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_shared_params(shared_params)

    private_emb_path = emb_cache / f"client_{cid}_private_emb.pt"
    if private_emb_path.exists():
        emb_state = torch.load(private_emb_path, map_location="cpu", weights_only=True)
        state = model.state_dict()
        for k, v in emb_state.items():
            if k in state and state[k].shape == v.shape:
                state[k] = v
        model.load_state_dict(state)
    model.eval()
    return model


def _post_fl_evaluation_hybrid(shared_params, cfg, data_dir):
    """
    Оценка финальной FL-модели для гибридного сценария.

    Если есть global_test.pkl — используем его (единый test для centralized и FL).
    RMSE/MAE: на всех триплетах global_test (публ. + прив. через модель владельца).
    Ranking (HR/NDCG): на приватных триплетах global_test, per-client.
    """
    info = load_global_info(data_dir)
    emb_cache = Path(data_dir) / "embed_cache"

    global_test_path = Path(data_dir) / "global_test.pkl"
    if not global_test_path.exists():
        return _post_fl_evaluation_hybrid_legacy(shared_params, cfg, data_dir)

    with open(global_test_path, "rb") as f:
        gt = pickle.load(f)
    pub_test = gt["public"]
    priv_test = gt["private"]

    print(f"\n{'='*55}")
    print(f"Оценка FL-модели на global_test "
          f"({len(pub_test)} публ. + {len(priv_test)} прив.)")
    print(f"{'='*55}")

    # Mapping: global user_id → (client_id, local private_user_id)
    user_to_client = {}
    for cid in range(info["num_clients"]):
        data = load_client_data(cid, data_dir)
        for global_uid, local_pid in data["private_user_mapping"].items():
            user_to_client[global_uid] = (cid, local_pid)

    # Public: оцениваем через модель клиента 0 (public embeddings одинаковые у всех)
    data0 = load_client_data(0, data_dir)
    model0 = _build_hybrid_model(data0, shared_params, emb_cache, 0, cfg)

    with torch.no_grad():
        users_t = torch.tensor(pub_test["public_user_id"].values.astype(np.int64),
                                dtype=torch.long)
        items_t = torch.tensor(pub_test["item_id"].values.astype(np.int64),
                                dtype=torch.long)
        is_pub_t = torch.ones(len(pub_test), dtype=torch.bool)
        pub_preds = model0(users_t, items_t, is_pub_t).numpy()
    pub_ratings = pub_test["rating"].values.astype(np.float32)

    # Private: для каждого клиента — его триплеты из priv_test
    priv_preds_all = []
    priv_ratings_all = []
    all_hr5, all_ndcg5, all_hr10, all_ndcg10 = [], [], [], []

    for cid in range(info["num_clients"]):
        data = load_client_data(cid, data_dir)
        client_priv_users = set(data["private_user_mapping"].keys())

        subset = priv_test[priv_test["user_id"].isin(client_priv_users)].copy()
        if len(subset) == 0:
            continue

        subset["private_user_id"] = subset["user_id"].map(data["private_user_mapping"])
        model = _build_hybrid_model(data, shared_params, emb_cache, cid, cfg)

        with torch.no_grad():
            u_t = torch.tensor(subset["private_user_id"].values.astype(np.int64),
                                dtype=torch.long)
            i_t = torch.tensor(subset["item_id"].values.astype(np.int64),
                                dtype=torch.long)
            is_pub_t = torch.zeros(len(subset), dtype=torch.bool)
            preds = model(u_t, i_t, is_pub_t).numpy()

        priv_preds_all.append(preds)
        priv_ratings_all.append(subset["rating"].values.astype(np.float32))

        # Ranking на приватных триплетах этого клиента
        test_pairs = list(zip(
            subset["private_user_id"].astype(int),
            subset["item_id"].astype(int),
        ))
        all_items_arr = np.arange(data["num_items"])

        class _PrivateWrapper:
            def __init__(self, m):
                self.m = m
            def __call__(self, u_t, i_t):
                is_pub = torch.zeros(len(u_t), dtype=torch.bool)
                return self.m(u_t, i_t, is_pub)
            def eval(self):
                self.m.eval()

        wrapper = _PrivateWrapper(model)
        hr5 = hit_rate_at_k(wrapper, test_pairs, all_items_arr, k=5, n_neg=99)
        ndcg5 = ndcg_at_k(wrapper, test_pairs, all_items_arr, k=5, n_neg=99)
        hr10 = hit_rate_at_k(wrapper, test_pairs, all_items_arr, k=10, n_neg=99)
        ndcg10 = ndcg_at_k(wrapper, test_pairs, all_items_arr, k=10, n_neg=99)

        all_hr5.append(hr5)
        all_ndcg5.append(ndcg5)
        all_hr10.append(hr10)
        all_ndcg10.append(ndcg10)

        print(f"  Клиент {cid:2d}: {len(subset):4d} прив. триплетов, HR@10={hr10:.3f}")

    # Общие RMSE/MAE на всём global_test
    all_preds = np.concatenate([pub_preds] + priv_preds_all) if priv_preds_all else pub_preds
    all_ratings = np.concatenate([pub_ratings] + priv_ratings_all) if priv_ratings_all else pub_ratings
    total_rmse = rmse(all_preds, all_ratings)
    total_mae = mae(all_preds, all_ratings)

    results = {
        "test_rmse": float(total_rmse),
        "test_mae": float(total_mae),
        "test_hr5": float(np.mean(all_hr5)) if all_hr5 else 0.0,
        "test_ndcg5": float(np.mean(all_ndcg5)) if all_ndcg5 else 0.0,
        "test_hr10": float(np.mean(all_hr10)) if all_hr10 else 0.0,
        "test_ndcg10": float(np.mean(all_ndcg10)) if all_ndcg10 else 0.0,
        "per_client_hr10": all_hr10,
        "per_client_rmse": [],  # больше не per-client, т.к. тест общий
    }

    print(f"\n  Global test:")
    print(f"    RMSE    = {results['test_rmse']:.4f}")
    print(f"    MAE     = {results['test_mae']:.4f}")
    print(f"    HR@5    = {results['test_hr5']:.4f}")
    print(f"    NDCG@5  = {results['test_ndcg5']:.4f}")
    print(f"    HR@10   = {results['test_hr10']:.4f}")
    print(f"    NDCG@10 = {results['test_ndcg10']:.4f}")

    return results


def _post_fl_evaluation_hybrid_legacy(shared_params, cfg, data_dir):
    """Старый per-client test (fallback если global_test.pkl отсутствует)."""
    info = load_global_info(data_dir)
    emb_cache = Path(data_dir) / "embed_cache"

    all_rmse, all_mae = [], []
    all_hr5, all_ndcg5 = [], []
    all_hr10, all_ndcg10 = [], []

    print(f"\n{'='*55}")
    print("Оценка FL-модели на per-client test (legacy)")
    print(f"{'='*55}")

    for cid in range(info["num_clients"]):
        data = load_client_data(cid, data_dir)
        model = _build_hybrid_model(data, shared_params, emb_cache, cid, cfg)
        test_df = data["test"]

        with torch.no_grad():
            is_public = test_df["is_public"].values.astype(bool)
            users = np.where(
                is_public,
                test_df["public_user_id"].fillna(0).values,
                test_df["private_user_id"].fillna(0).values,
            ).astype(np.int64)
            items = test_df["item_id"].values
            ratings = test_df["rating"].values.astype(np.float32)

            users_t = torch.tensor(users, dtype=torch.long)
            items_t = torch.tensor(items, dtype=torch.long)
            is_pub_t = torch.tensor(is_public, dtype=torch.bool)
            preds = model(users_t, items_t, is_pub_t).numpy()

        c_rmse = rmse(preds, ratings)
        c_mae = mae(preds, ratings)
        all_rmse.append(c_rmse)
        all_mae.append(c_mae)

        test_private = test_df[test_df["is_public"] == 0].copy()
        if len(test_private) > 0:
            test_pairs = list(zip(
                test_private["private_user_id"].fillna(0).astype(int),
                test_private["item_id"].astype(int),
            ))
            all_items_arr = np.arange(data["num_items"])

            class _PrivateWrapper:
                def __init__(self, m):
                    self.m = m
                def __call__(self, u_t, i_t):
                    is_pub = torch.zeros(len(u_t), dtype=torch.bool)
                    return self.m(u_t, i_t, is_pub)
                def eval(self):
                    self.m.eval()

            wrapper = _PrivateWrapper(model)
            hr5 = hit_rate_at_k(wrapper, test_pairs, all_items_arr, k=5, n_neg=99)
            ndcg5 = ndcg_at_k(wrapper, test_pairs, all_items_arr, k=5, n_neg=99)
            hr10 = hit_rate_at_k(wrapper, test_pairs, all_items_arr, k=10, n_neg=99)
            ndcg10 = ndcg_at_k(wrapper, test_pairs, all_items_arr, k=10, n_neg=99)
        else:
            hr5 = ndcg5 = hr10 = ndcg10 = 0.0

        all_hr5.append(hr5)
        all_ndcg5.append(ndcg5)
        all_hr10.append(hr10)
        all_ndcg10.append(ndcg10)

        print(f"  Клиент {cid:2d}: RMSE={c_rmse:.4f}  HR@10={hr10:.3f}")

    results = {
        "test_rmse": float(np.mean(all_rmse)),
        "test_mae": float(np.mean(all_mae)),
        "test_hr5": float(np.mean(all_hr5)),
        "test_ndcg5": float(np.mean(all_ndcg5)),
        "test_hr10": float(np.mean(all_hr10)),
        "test_ndcg10": float(np.mean(all_ndcg10)),
        "per_client_rmse": all_rmse,
        "per_client_hr10": all_hr10,
    }

    print(f"\n  Среднее по клиентам:")
    print(f"    RMSE    = {results['test_rmse']:.4f}")
    print(f"    MAE     = {results['test_mae']:.4f}")
    print(f"    HR@5    = {results['test_hr5']:.4f}")
    print(f"    NDCG@5  = {results['test_ndcg5']:.4f}")
    print(f"    HR@10   = {results['test_hr10']:.4f}")
    print(f"    NDCG@10 = {results['test_ndcg10']:.4f}")

    return results


def run_simulation(cfg, data_dir="data/processed"):
    """
    Запуск FL-симуляции через Flower.

    Автоматически определяет режим (hybrid или обычный non-IID)
    по наличию флага is_hybrid в global_info.
    """
    info = load_global_info(data_dir)
    n_clients = info["num_clients"]
    cfg["num_clients"] = n_clients
    is_hybrid = info.get("is_hybrid", False)

    # Чистим кеш эмбеддингов от прошлого запуска
    emb_cache = Path(data_dir) / "embed_cache"
    if emb_cache.exists():
        shutil.rmtree(emb_cache)

    mode = "гибридный" if is_hybrid else "non-IID"
    mu = cfg.get("proximal_mu", 0.0)
    strategy = "FedProx" if mu > 0 else "FedAvg"
    dp_cfg = cfg.get("dp", {})

    print(f"Режим:    {mode}")
    print(f"Стратегия: {strategy}" + (f" (mu={mu})" if mu > 0 else ""))
    if dp_cfg.get("enabled"):
        print(f"DP:       C={dp_cfg['max_grad_norm']}, sigma={dp_cfg['noise_multiplier']}")
    print(f"Клиентов: {n_clients}")
    print(f"Раундов:  {cfg['num_rounds']}")
    print(f"Fit:      {cfg['fraction_fit']*100:.0f}% клиентов за раунд")

    # Начальные shared-параметры
    if is_hybrid:
        # Гибридный режим: публичные + приватные юзеры
        ref_model = HybridNCF(
            num_public_users=info["num_public_users"],
            num_private_users=10,  # Любое значение, нам нужны только shared params
            num_items=info["total_items"],
            emb_dim=cfg["emb_dim"],
            mlp_layers=cfg.get("mlp_layers")
        )
        client_fn = make_hybrid_client_fn(cfg, data_dir)
        eval_fn = _post_fl_evaluation_hybrid
    else:
        # Обычный non-IID режим
        ref_model = NCF(
            num_users=10, num_items=10,
            emb_dim=cfg["emb_dim"],
            mlp_layers=cfg.get("mlp_layers")
        )
        client_fn = make_client_fn(cfg, data_dir)
        eval_fn = _post_fl_evaluation

    init_params = fl.common.ndarrays_to_parameters(ref_model.get_shared_params())

    # Путь для сохранения финальных агрегированных параметров
    params_path = Path(data_dir) / "fl_shared_params.pkl"

    strategy = _SavingFedAvg(
        save_path=str(params_path),
        fraction_fit=cfg["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=max(2, int(n_clients * cfg["fraction_fit"])),
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=_weighted_avg,
        evaluate_metrics_aggregation_fn=_weighted_avg,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "include_dashboard": False,
            "log_to_driver": False,
            "logging_level": "ERROR",
        },
    )

    # Privacy accounting
    privacy_info = {}
    if dp_cfg.get("enabled"):
        accountant = PrivacyAccountant(
            dp_cfg["noise_multiplier"],
            dp_cfg.get("target_delta", 1e-5),
        )
        # Каждый клиент участвует в каждом раунде с вероятностью fraction_fit
        # Считаем worst-case: клиент участвует во всех раундах
        for _ in range(cfg["num_rounds"]):
            accountant.step()
        eps = accountant.get_epsilon()
        eps_adv = accountant.get_epsilon_advanced()
        delta = dp_cfg.get("target_delta", 1e-5)
        print(f"\n  Приватность (simple composition):   eps={eps:.2f}, delta={delta}")
        print(f"  Приватность (advanced composition): eps={eps_adv:.2f}, delta={delta}")
        privacy_info = {
            "dp_epsilon": eps,
            "dp_epsilon_advanced": eps_adv,
            "dp_delta": delta,
            "dp_sigma": dp_cfg["noise_multiplier"],
            "dp_max_grad_norm": dp_cfg["max_grad_norm"],
        }

    # Пост-FL оценка на тестовых данных
    post_eval = {}
    if params_path.exists():
        with open(params_path, "rb") as f:
            final_shared_params = pickle.load(f)
        post_eval = eval_fn(final_shared_params, cfg, data_dir)

    # Сохраняем полную историю
    out_path = Path(data_dir) / "fl_history.pkl"
    fl_data = {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed": history.metrics_distributed,
        "metrics_centralized": history.metrics_centralized,
        "metrics_distributed_fit": getattr(history, "metrics_distributed_fit", {}),
    }
    fl_data.update(post_eval)
    fl_data.update(privacy_info)

    with open(out_path, "wb") as f:
        pickle.dump(fl_data, f)
    print(f"\nИстория сохранена: {out_path}")

    return history
