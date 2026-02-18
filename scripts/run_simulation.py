#!/usr/bin/env python3
"""
Запуск FL-симуляции: 20 клиентов с приватными данными,
FedAvg агрегация только shared-параметров (MLP + output).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.federated.server import run_simulation


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    tc = cfg["training"]
    fc = cfg["federated"]

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
    }

    print("=" * 55)
    print("Федеративное обучение — симуляция")
    print("=" * 55)

    run_simulation(sim_cfg)

    print("\nГотово.")


if __name__ == "__main__":
    main()
