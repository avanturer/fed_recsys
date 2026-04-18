#!/usr/bin/env python3
"""
Дополнительные эксперименты:
- Перегон централизованного baseline с early stopping
- DP с большими sigma (1.0, 2.0) для меaningful privacy budget

Запускать после основного run_experiments.py.
"""
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


def save_results(experiment_name, dataset_name):
    out = RESULTS_DIR / dataset_name / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    for fname in ["centralized_history.pkl", "centralized_model.pt",
                  "fl_history.pkl", "fl_shared_params.pkl", "global_info.pkl"]:
        src = DATA_DIR / fname
        if src.exists():
            shutil.copy2(src, out / fname)
    print(f"  Сохранено: {out}")


def run_centralized_for_dataset(ds_name, cfg):
    """Перегон centralized с early stopping."""
    print(f"\n=== Перегон centralized: {ds_name} ===")
    save_config(cfg, "configs/config.yaml")

    # Reimport чтобы обновить конфиг
    import importlib
    import scripts.prepare_data as prep_mod
    import scripts.train_centralized as train_mod
    importlib.reload(prep_mod)
    importlib.reload(train_mod)

    prep_mod.main()
    train_mod.main()
    save_results("centralized", ds_name)


def run_fl(ds_name, cfg, exp_name):
    print(f"\n=== {exp_name} на {ds_name} ===")
    save_config(cfg, "configs/config.yaml")

    import importlib
    import scripts.run_simulation as sim_mod
    importlib.reload(sim_mod)
    sim_mod.main()
    save_results(exp_name, ds_name)


def main():
    base = load_config()
    original = yaml.safe_load(yaml.dump(base))

    try:
        # Конфиги для каждого датасета
        ml_cfg = yaml.safe_load(yaml.dump(base))
        ml_cfg["data"]["dataset"] = "ml-1m"
        ml_cfg["data"]["num_clients"] = 20
        ml_cfg["model"]["embedding_dim"] = 64
        ml_cfg["training"]["local_epochs"] = 2

        am_cfg = yaml.safe_load(yaml.dump(base))
        am_cfg["data"]["dataset"] = "amazon-music"
        am_cfg["data"]["num_clients"] = 10
        am_cfg["model"]["embedding_dim"] = 32
        am_cfg["training"]["local_epochs"] = 3

        # 1. Перегон centralized (теперь с early stopping)
        run_centralized_for_dataset("ml-1m", ml_cfg)
        run_centralized_for_dataset("amazon-music", am_cfg)

        # 2. DP с большими sigma — по 2 эксперимента на каждый датасет
        for ds_name, cfg in [("ml-1m", ml_cfg), ("amazon-music", am_cfg)]:
            # Для FL экспериментов данные уже подготовлены centralized-этапом
            save_config(cfg, "configs/config.yaml")
            import importlib
            import scripts.prepare_data as prep_mod
            importlib.reload(prep_mod)
            prep_mod.main()

            for sigma, tag in [(1.0, "s10"), (2.0, "s20")]:
                cfg_dp = yaml.safe_load(yaml.dump(cfg))
                cfg_dp["federated"]["strategy"] = "fedavg"
                cfg_dp["federated"]["proximal_mu"] = 0.0
                cfg_dp["federated"]["dp"]["enabled"] = True
                cfg_dp["federated"]["dp"]["noise_multiplier"] = sigma
                run_fl(ds_name, cfg_dp, f"fedavg_dp_{tag}")

        print("\n\nДополнительные эксперименты завершены.")
    finally:
        save_config(original, "configs/config.yaml")


if __name__ == "__main__":
    main()
