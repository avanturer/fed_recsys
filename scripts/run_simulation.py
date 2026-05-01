#!/usr/bin/env python3
"""
Запуск FL-симуляции: клиенты с приватными данными,
FedAvg/FedProx агрегация только shared-параметров.
Опционально: дифференциальная приватность.
"""
import os
# Гасим телеметрию Ray до импорта — иначе env vars уже не подхватятся.
# На WSL2 без этого GCS копит task events и через 15-20 минут падает.
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("RAY_event_stats", "0")
os.environ.setdefault("RAY_record_ref_creation_sites", "0")
os.environ.setdefault("RAY_record_task_actor_creation_sites", "0")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch

from src.federated.server import run_simulation


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Сидируем всё для воспроизводимости
    seed = cfg["data"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    mc = cfg["model"]
    tc = cfg["training"]
    fc = cfg["federated"]

    # Определяем mu: для fedprox берём из конфига, для fedavg — 0
    strategy = fc.get("strategy", "fedavg")
    mu = fc.get("proximal_mu", 0.01) if strategy == "fedprox" else 0.0

    sim_cfg = {
        "num_rounds": fc["num_rounds"],
        "fraction_fit": fc["fraction_fit"],
        "emb_dim": mc["embedding_dim"],
        "mlp_layers": mc["mlp_layers"],
        "dropout": mc["dropout"],
        "batch_size": tc["batch_size"],
        "local_epochs": tc["local_epochs"],
        "lr": tc["learning_rate"],
        "weight_decay": fc.get("weight_decay", tc.get("weight_decay", 0)),
        "proximal_mu": mu,
        "dp": fc.get("dp", {}),
        "seed": seed,
    }

    print("=" * 55)
    print("Федеративное обучение — симуляция")
    print("=" * 55)
    print(f"Стратегия: {strategy}" + (f" (mu={mu})" if mu > 0 else ""))
    dp_cfg = fc.get("dp", {})
    if dp_cfg.get("enabled"):
        print(f"DP:        C={dp_cfg['max_grad_norm']}, "
              f"sigma={dp_cfg['noise_multiplier']}")

    run_simulation(sim_cfg)

    print("\nГотово.")


if __name__ == "__main__":
    main()
