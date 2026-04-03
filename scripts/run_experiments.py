#!/usr/bin/env python3
"""
Полный прогон экспериментов для ВКР.

Для каждого датасета прогоняет:
1. Централизованный baseline (обычный NCF)
2. FedAvg (гибридный)
3. FedProx (гибридный)
4. FedAvg + DP
5. FedProx + DP

Результаты сохраняются в data/results/<dataset>/<experiment>/
"""
import os
import sys
import shutil
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def run_prepare(cfg):
    """Подготовка данных."""
    save_config(cfg, "configs/config.yaml")
    from scripts.prepare_data import main as prepare_main
    prepare_main()


def run_centralized(cfg):
    """Централизованный baseline."""
    save_config(cfg, "configs/config.yaml")
    from scripts.train_centralized import main as centralized_main
    centralized_main()


def run_fl(cfg):
    """FL-симуляция."""
    save_config(cfg, "configs/config.yaml")

    # Нужно перезагрузить модуль, т.к. конфиг читается при вызове
    from scripts.run_simulation import main as sim_main
    sim_main()


def save_results(experiment_name, dataset_name):
    """Копируем результаты в отдельную папку."""
    out = RESULTS_DIR / dataset_name / experiment_name
    out.mkdir(parents=True, exist_ok=True)

    for fname in ["centralized_history.pkl", "centralized_model.pt",
                   "fl_history.pkl", "fl_shared_params.pkl", "global_info.pkl"]:
        src = DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    print(f"  Результаты: {out}")


def print_metrics(path, label):
    """Выводим ключевые метрики из сохранённого файла."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        data = pickle.load(f)

    rmse = data.get("test_rmse", data.get("ranking", {}).get("rmse", "?"))
    mae = data.get("test_mae", "?")
    hr10 = data.get("test_hr10", data.get("ranking", {}).get("hr@10", "?"))
    ndcg10 = data.get("test_ndcg10", data.get("ranking", {}).get("ndcg@10", "?"))

    print(f"  {label}: RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"HR@10={hr10:.4f}  NDCG@10={ndcg10:.4f}")


def run_dataset_experiments(dataset_name, base_cfg):
    """Все эксперименты для одного датасета."""
    cfg = base_cfg.copy()

    print(f"\n{'='*60}")
    print(f"ДАТАСЕТ: {dataset_name}")
    print(f"{'='*60}")

    # 1. Подготовка данных
    print("\n--- Подготовка данных ---")
    run_prepare(cfg)

    # 2. Централизованный baseline
    print("\n--- Централизованный baseline ---")
    run_centralized(cfg)
    save_results("centralized", dataset_name)

    # 3. FedAvg
    print("\n--- FedAvg ---")
    cfg["federated"]["strategy"] = "fedavg"
    cfg["federated"]["proximal_mu"] = 0.0
    cfg["federated"]["dp"]["enabled"] = False
    run_fl(cfg)
    save_results("fedavg", dataset_name)

    # 4. FedProx
    print("\n--- FedProx (mu=0.01) ---")
    cfg["federated"]["strategy"] = "fedprox"
    cfg["federated"]["proximal_mu"] = 0.01
    cfg["federated"]["dp"]["enabled"] = False
    run_fl(cfg)
    save_results("fedprox", dataset_name)

    # 5. FedAvg + DP
    print("\n--- FedAvg + DP ---")
    cfg["federated"]["strategy"] = "fedavg"
    cfg["federated"]["proximal_mu"] = 0.0
    cfg["federated"]["dp"]["enabled"] = True
    run_fl(cfg)
    save_results("fedavg_dp", dataset_name)

    # 6. FedProx + DP
    print("\n--- FedProx + DP ---")
    cfg["federated"]["strategy"] = "fedprox"
    cfg["federated"]["proximal_mu"] = 0.01
    cfg["federated"]["dp"]["enabled"] = True
    run_fl(cfg)
    save_results("fedprox_dp", dataset_name)

    # Итоговая таблица
    print(f"\n{'='*60}")
    print(f"ИТОГИ: {dataset_name}")
    print(f"{'='*60}")
    res = RESULTS_DIR / dataset_name
    print_metrics(res / "centralized" / "centralized_history.pkl", "Centralized")
    for exp in ["fedavg", "fedprox", "fedavg_dp", "fedprox_dp"]:
        print_metrics(res / exp / "fl_history.pkl", exp.upper().replace("_", "+"))


def main():
    base_cfg = load_config()
    # Сохраняем оригинальный конфиг для восстановления после всех экспериментов
    original_cfg = yaml.safe_load(yaml.dump(base_cfg))

    try:
        # MovieLens-1M
        ml_cfg = yaml.safe_load(yaml.dump(base_cfg))
        ml_cfg["data"]["dataset"] = "ml-1m"
        ml_cfg["data"]["num_clients"] = 20
        run_dataset_experiments("ml-1m", ml_cfg)

        # Amazon Digital Music
        am_cfg = yaml.safe_load(yaml.dump(base_cfg))
        am_cfg["data"]["dataset"] = "amazon-music"
        am_cfg["data"]["num_clients"] = 10
        am_cfg["model"]["embedding_dim"] = 32
        am_cfg["training"]["local_epochs"] = 3
        run_dataset_experiments("amazon-music", am_cfg)

        print("\n\nВсе эксперименты завершены.")
    finally:
        # Восстанавливаем оригинальный конфиг
        save_config(original_cfg, "configs/config.yaml")


if __name__ == "__main__":
    main()
