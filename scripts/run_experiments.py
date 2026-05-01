#!/usr/bin/env python3
"""
Прогоняет все эксперименты по двум датасетам и собирает таблицу метрик.

Каждый запуск train_centralized / run_simulation запускается в отдельном
subprocess: иначе Ray в одном процессе копит task events и через 15-20
минут на WSL2 GCS отваливается. Если результат уже есть в data/results/,
этап просто пропускается — можно перезапускать сколько угодно.
"""
import sys
import shutil
import pickle
import subprocess
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# (имя_папки, strategy, mu, dp_enabled, sigma)
FL_EXPERIMENTS = [
    ("fedavg",          "fedavg",  0.0,  False, 0.0),
    ("fedprox",         "fedprox", 0.01, False, 0.0),
    ("fedavg_dp_s01",   "fedavg",  0.0,  True,  0.1),
    ("fedavg_dp_s05",   "fedavg",  0.0,  True,  0.5),
    ("fedavg_dp_s10",   "fedavg",  0.0,  True,  1.0),
    ("fedavg_dp_s20",   "fedavg",  0.0,  True,  2.0),
    ("fedprox_dp_s01",  "fedprox", 0.01, True,  0.1),
    ("fedprox_dp_s05",  "fedprox", 0.01, True,  0.5),
]

TABLE_LAYOUT = [
    ("Centralized",       "centralized",     "centralized_history.pkl"),
    ("FedAvg",            "fedavg",          "fl_history.pkl"),
    ("FedProx (mu=0.01)", "fedprox",         "fl_history.pkl"),
    ("FedAvg+DP s=0.1",   "fedavg_dp_s01",   "fl_history.pkl"),
    ("FedAvg+DP s=0.5",   "fedavg_dp_s05",   "fl_history.pkl"),
    ("FedAvg+DP s=1.0",   "fedavg_dp_s10",   "fl_history.pkl"),
    ("FedAvg+DP s=2.0",   "fedavg_dp_s20",   "fl_history.pkl"),
    ("FedProx+DP s=0.1",  "fedprox_dp_s01",  "fl_history.pkl"),
    ("FedProx+DP s=0.5",  "fedprox_dp_s05",  "fl_history.pkl"),
]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def write_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def run_script(script_name):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script_name}.py")]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"{script_name} упал с кодом {res.returncode}")


def already_done(experiment, dataset):
    out = RESULTS_DIR / dataset / experiment
    if not out.exists():
        return False
    if experiment == "centralized":
        return (out / "centralized_history.pkl").exists()
    return (out / "fl_history.pkl").exists()


def save_results(experiment, dataset):
    out = RESULTS_DIR / dataset / experiment
    out.mkdir(parents=True, exist_ok=True)
    artefacts = [
        "centralized_history.pkl", "centralized_model.pt",
        "fl_history.pkl", "fl_shared_params.pkl",
        "global_info.pkl", "global_test.pkl",
    ]
    for name in artefacts:
        src = DATA_DIR / name
        if src.exists():
            shutil.copy2(src, out / name)
    print(f"  -> {out}")


def run_dataset(dataset_name, cfg):
    print(f"\n{'='*65}\n  {dataset_name.upper()}\n{'='*65}")

    pending = []
    if not already_done("centralized", dataset_name):
        pending.append("centralized")
    for exp_name, *_ in FL_EXPERIMENTS:
        if not already_done(exp_name, dataset_name):
            pending.append(exp_name)

    if not pending:
        print("  Все эксперименты уже посчитаны, skip.")
        return

    print(f"  Нужно посчитать: {', '.join(pending)}")

    print("\n--- Подготовка данных ---")
    write_config(cfg)
    run_script("prepare_data")

    if "centralized" in pending:
        print("\n--- Централизованный baseline ---")
        write_config(cfg)
        run_script("train_centralized")
        save_results("centralized", dataset_name)

    for exp_name, strategy, mu, dp_enabled, sigma in FL_EXPERIMENTS:
        if exp_name not in pending:
            continue

        title = strategy.upper()
        if mu > 0:
            title += f" mu={mu}"
        if dp_enabled:
            title += f" + DP sigma={sigma}"
        print(f"\n--- {title} ---")

        cfg_exp = copy.deepcopy(cfg)
        cfg_exp["federated"]["strategy"] = strategy
        cfg_exp["federated"]["proximal_mu"] = mu
        cfg_exp["federated"]["dp"]["enabled"] = dp_enabled
        if dp_enabled:
            cfg_exp["federated"]["dp"]["noise_multiplier"] = sigma

        write_config(cfg_exp)
        run_script("run_simulation")
        save_results(exp_name, dataset_name)


def aggregate(dataset_name):
    print(f"\n{'='*78}\n  ИТОГИ: {dataset_name.upper()}\n{'='*78}")
    header = f"  {'Метод':<20} {'RMSE':>10} {'MAE':>10} {'HR@10':>10} {'NDCG@10':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, folder, fname in TABLE_LAYOUT:
        path = RESULTS_DIR / dataset_name / folder / fname
        if not path.exists():
            print(f"  {label:<20} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        rk = data.get("ranking", {})
        rmse = data.get("test_rmse", rk.get("rmse", float("nan")))
        mae = data.get("test_mae", rk.get("mae", float("nan")))
        hr = data.get("test_hr10", rk.get("hr@10", float("nan")))
        ndcg = data.get("test_ndcg10", rk.get("ndcg@10", float("nan")))
        print(f"  {label:<20} {rmse:>10.4f} {mae:>10.4f} {hr:>10.4f} {ndcg:>10.4f}")


def main():
    # Сохраняем оригинал как текст, иначе yaml.dump убьёт комментарии и порядок
    original_text = CONFIG_PATH.read_text()
    base_cfg = load_config()

    try:
        ml_cfg = copy.deepcopy(base_cfg)
        ml_cfg["data"]["dataset"] = "ml-1m"
        ml_cfg["data"]["num_clients"] = 20

        # Amazon Digital Music разрежён (1.29 отзыва на юзера),
        # поэтому 2-core, маленькие embedding и сильнее регуляризация
        am_cfg = copy.deepcopy(base_cfg)
        am_cfg["data"]["dataset"] = "amazon-music"
        am_cfg["data"]["num_clients"] = 10
        am_cfg["data"]["min_user_ratings"] = 2
        am_cfg["data"]["min_item_ratings"] = 2
        am_cfg["model"]["embedding_dim"] = 16
        am_cfg["model"]["mlp_layers"] = [64, 32]
        am_cfg["model"]["dropout"] = 0.3
        am_cfg["training"]["local_epochs"] = 3
        am_cfg["training"]["weight_decay"] = 0.01

        run_dataset("ml-1m", ml_cfg)
        run_dataset("amazon-music", am_cfg)

        aggregate("ml-1m")
        aggregate("amazon-music")

        print("\n\nВсе эксперименты готовы. Графики: python scripts/generate_plots.py")
    finally:
        CONFIG_PATH.write_text(original_text)


if __name__ == "__main__":
    main()
