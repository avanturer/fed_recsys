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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")

# Общий счётчик для прогресса
TOTAL_STEPS = 16  # 2 датасета * (prepare + centralized + 2 FL + 4 DP)
_current_step = 0
_start_time = None


def _progress(label):
    """Выводит прогресс-бар с номером шага и прошедшим временем."""
    global _current_step, _start_time
    if _start_time is None:
        _start_time = time.time()
    _current_step += 1
    elapsed = time.time() - _start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    bar_len = 30
    filled = int(bar_len * _current_step / TOTAL_STEPS)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = _current_step / TOTAL_STEPS * 100
    print(f"\n{'='*60}")
    print(f"  [{bar}] {pct:.0f}%  ({_current_step}/{TOTAL_STEPS})  {mins:02d}:{secs:02d}")
    print(f"  >>> {label}")
    print(f"{'='*60}")
    sys.stdout.flush()


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

    # 1. Подготовка данных
    _progress(f"{dataset_name} — подготовка данных")
    run_prepare(cfg)

    # 2. Централизованный baseline
    _progress(f"{dataset_name} — централизованный baseline")
    run_centralized(cfg)
    save_results("centralized", dataset_name)

    # 3. FedAvg
    _progress(f"{dataset_name} — FedAvg")
    cfg["federated"]["strategy"] = "fedavg"
    cfg["federated"]["proximal_mu"] = 0.0
    cfg["federated"]["dp"]["enabled"] = False
    run_fl(cfg)
    save_results("fedavg", dataset_name)

    # 4. FedProx
    _progress(f"{dataset_name} — FedProx (mu=0.01)")
    cfg["federated"]["strategy"] = "fedprox"
    cfg["federated"]["proximal_mu"] = 0.01
    cfg["federated"]["dp"]["enabled"] = False
    run_fl(cfg)
    save_results("fedprox", dataset_name)

    # DP-ablation: два режима шума
    dp_configs = [
        ("dp_s05", 0.5, 1.0),   # умеренный шум
        ("dp_s01", 0.1, 1.0),   # сильный шум
    ]

    for dp_tag, sigma, clip_norm in dp_configs:
        # FedAvg + DP
        _progress(f"{dataset_name} — FedAvg + DP (sigma={sigma})")
        cfg["federated"]["strategy"] = "fedavg"
        cfg["federated"]["proximal_mu"] = 0.0
        cfg["federated"]["dp"]["enabled"] = True
        cfg["federated"]["dp"]["noise_multiplier"] = sigma
        cfg["federated"]["dp"]["max_grad_norm"] = clip_norm
        run_fl(cfg)
        save_results(f"fedavg_{dp_tag}", dataset_name)

        # FedProx + DP
        _progress(f"{dataset_name} — FedProx + DP (sigma={sigma})")
        cfg["federated"]["strategy"] = "fedprox"
        cfg["federated"]["proximal_mu"] = 0.01
        cfg["federated"]["dp"]["enabled"] = True
        run_fl(cfg)
        save_results(f"fedprox_{dp_tag}", dataset_name)

    # Итоговая таблица
    print(f"\n{'='*60}")
    print(f"ИТОГИ: {dataset_name}")
    print(f"{'='*60}")
    res = RESULTS_DIR / dataset_name
    print_metrics(res / "centralized" / "centralized_history.pkl", "Centralized")
    experiments = ["fedavg", "fedprox"]
    for dp_tag, sigma, _ in dp_configs:
        experiments += [f"fedavg_{dp_tag}", f"fedprox_{dp_tag}"]
    for exp in experiments:
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

        # Amazon Digital Music (разрежённый — усиленная регуляризация)
        am_cfg = yaml.safe_load(yaml.dump(base_cfg))
        am_cfg["data"]["dataset"] = "amazon-music"
        am_cfg["data"]["num_clients"] = 10
        am_cfg["data"]["min_user_ratings"] = 2
        am_cfg["data"]["min_item_ratings"] = 2
        am_cfg["model"]["embedding_dim"] = 16
        am_cfg["model"]["mlp_layers"] = [64, 32]
        am_cfg["model"]["dropout"] = 0.3
        am_cfg["training"]["local_epochs"] = 3
        am_cfg["training"]["weight_decay"] = 0.01
        run_dataset_experiments("amazon-music", am_cfg)

        elapsed = time.time() - _start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"\n\n{'='*60}")
        print(f"  Все эксперименты завершены за {mins:02d}:{secs:02d}")
        print(f"{'='*60}")
    finally:
        # Восстанавливаем оригинальный конфиг
        save_config(original_cfg, "configs/config.yaml")


if __name__ == "__main__":
    main()
