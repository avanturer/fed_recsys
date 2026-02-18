"""
Flower-клиент для федеративного обучения рекомендательной системы.

Ключевая особенность: агрегируем только shared-параметры (MLP + output),
а эмбеддинги юзеров/айтемов остаются локальными на каждом клиенте,
потому что у разных клиентов разные юзеры и разные локальные ID.

Эмбеддинги кешируются на диск между раундами, чтобы накапливать знания.
"""
import os

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.ncf import NCF, HybridNCF
from src.data.splitter import ClientDataset, HybridClientDataset, load_client_data


class RecClient(fl.client.NumPyClient):
    """
    Клиент = одна компания с приватными данными.
    На сервер уходят только веса MLP, обратно — тоже только MLP.
    Сырые данные никогда не покидают клиент.
    Эмбеддинги сохраняются на диск между раундами FL.
    """

    def __init__(self, client_id, model, train_loader, val_loader,
                 local_epochs=5, lr=0.001, weight_decay=0, device="cpu",
                 data_dir="data/processed"):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.device = device
        self.data_dir = data_dir

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.criterion = torch.nn.MSELoss()

        # Подгружаем эмбеддинги с прошлого раунда (если есть)
        self._load_cached_embeddings()

    def _emb_cache_dir(self):
        return os.path.join(self.data_dir, "embed_cache")

    def _emb_path(self):
        return os.path.join(self._emb_cache_dir(), f"client_{self.client_id}_emb.pt")

    def _load_cached_embeddings(self):
        """Загружаем эмбеддинги с прошлого раунда, если файл есть."""
        path = self._emb_path()
        if not os.path.exists(path):
            return
        emb_state = torch.load(path, map_location="cpu", weights_only=True)
        state = self.model.state_dict()
        for k, v in emb_state.items():
            if k in state and state[k].shape == v.shape:
                state[k] = v
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _save_embeddings(self):
        """Сохраняем эмбеддинги на диск для следующего раунда."""
        os.makedirs(self._emb_cache_dir(), exist_ok=True)
        state = self.model.state_dict()
        emb_keys = [k for k in state if "emb" in k]
        emb_state = {k: state[k].cpu() for k in emb_keys}
        torch.save(emb_state, self._emb_path())

    def get_parameters(self, config=None):
        """Отдаём серверу только shared-параметры (без эмбеддингов)."""
        return self.model.get_shared_params()

    def set_parameters(self, params):
        """Принимаем с сервера shared-параметры, эмбеддинги не трогаем."""
        self.model.load_shared_params(params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Сбрасываем optimizer state после загрузки агрегированных параметров,
        # чтобы Adam не использовал стухшие momentum/variance от прошлого раунда
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.optimizer.defaults["lr"],
            weight_decay=self.optimizer.defaults["weight_decay"],
        )

        epochs = config.get("local_epochs", self.local_epochs)
        total_loss = 0.0
        for _ in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in self.train_loader:
                u = batch["user"].to(self.device)
                i = batch["item"].to(self.device)
                r = batch["rating"].to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(u, i)
                loss = self.criterion(pred, r)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            total_loss += epoch_loss / max(n_batches, 1)

        avg_loss = total_loss / max(epochs, 1)
        n_samples = len(self.train_loader.dataset)

        # Сохраняем обученные эмбеддинги для следующего раунда
        self._save_embeddings()

        return self.get_parameters(), n_samples, {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                u = batch["user"].to(self.device)
                i = batch["item"].to(self.device)
                r = batch["rating"].to(self.device)

                pred = self.model(u, i)
                batch_size = len(r)
                total_loss += self.criterion(pred, r).item() * batch_size
                total_samples += batch_size
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(r.cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        rmse_val = float(np.sqrt(np.mean((preds - targets) ** 2)))

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, len(preds), {"rmse": rmse_val}


class HybridRecClient(fl.client.NumPyClient):
    """
    Клиент для гибридного сценария (публичные + приватные юзеры).

    На сервер отправляются:
    - Публичные user embeddings (общие для всех клиентов)
    - Item embeddings (все айтемы глобальные)
    - MLP + output

    Локальные (не агрегируются):
    - Приватные user embeddings (уникальные для каждого клиента)
    """

    def __init__(self, client_id, model, train_loader, val_loader,
                 local_epochs=5, lr=0.001, weight_decay=0, device="cpu",
                 data_dir="data/processed"):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.device = device
        self.data_dir = data_dir

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.criterion = torch.nn.MSELoss()

        # Загружаем ТОЛЬКО приватные эмбеддинги с прошлого раунда
        self._load_cached_private_embeddings()

    def _emb_cache_dir(self):
        return os.path.join(self.data_dir, "embed_cache")

    def _private_emb_path(self):
        return os.path.join(self._emb_cache_dir(), f"client_{self.client_id}_private_emb.pt")

    def _load_cached_private_embeddings(self):
        """Загружаем только приватные эмбеддинги с прошлого раунда."""
        path = self._private_emb_path()
        if not os.path.exists(path):
            return
        emb_state = torch.load(path, map_location="cpu", weights_only=True)
        state = self.model.state_dict()
        for k, v in emb_state.items():
            if k in state and state[k].shape == v.shape:
                state[k] = v
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _save_private_embeddings(self):
        """Сохраняем только приватные эмбеддинги на диск."""
        os.makedirs(self._emb_cache_dir(), exist_ok=True)
        state = self.model.state_dict()
        # Только приватные эмбеддинги
        private_keys = [k for k in state if "private_user_emb" in k]
        private_state = {k: state[k].cpu() for k in private_keys}
        torch.save(private_state, self._private_emb_path())

    def get_parameters(self, config=None):
        """Отдаём серверу shared-параметры (публичные эмбеддинги + айтемы + MLP + output)."""
        return self.model.get_shared_params()

    def set_parameters(self, params):
        """Принимаем с сервера shared-параметры, приватные эмбеддинги не трогаем."""
        self.model.load_shared_params(params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Сбрасываем optimizer state после загрузки агрегированных параметров
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.optimizer.defaults["lr"],
            weight_decay=self.optimizer.defaults["weight_decay"],
        )

        epochs = config.get("local_epochs", self.local_epochs)
        total_loss = 0.0
        for _ in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in self.train_loader:
                u = batch["user"].to(self.device)
                i = batch["item"].to(self.device)
                r = batch["rating"].to(self.device)
                is_pub = batch["is_public"].to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(u, i, is_pub)
                loss = self.criterion(pred, r)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            total_loss += epoch_loss / max(n_batches, 1)

        avg_loss = total_loss / max(epochs, 1)
        n_samples = len(self.train_loader.dataset)

        # Сохраняем приватные эмбеддинги
        self._save_private_embeddings()

        return self.get_parameters(), n_samples, {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                u = batch["user"].to(self.device)
                i = batch["item"].to(self.device)
                r = batch["rating"].to(self.device)
                is_pub = batch["is_public"].to(self.device)

                pred = self.model(u, i, is_pub)
                batch_size = len(r)
                total_loss += self.criterion(pred, r).item() * batch_size
                total_samples += batch_size
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(r.cpu().numpy())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        rmse_val = float(np.sqrt(np.mean((preds - targets) ** 2)))

        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss, len(preds), {"rmse": rmse_val}


def make_client_fn(cfg, data_dir="data/processed"):
    """
    Фабрика клиентов для Flower-симуляции.
    Каждый вызов client_fn(cid) создаёт нового клиента с его локальными данными.
    """
    def client_fn(cid):
        client_id = int(cid)
        data = load_client_data(client_id, data_dir)

        train_ds = ClientDataset(data["train"])
        val_ds = ClientDataset(data["val"])

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                                shuffle=False, num_workers=0)

        model = NCF(
            num_users=data["num_users"],
            num_items=data["num_items"],
            emb_dim=cfg["emb_dim"],
            mlp_layers=cfg.get("mlp_layers"),
            dropout=cfg.get("dropout", 0.2),
        )

        return RecClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=cfg["local_epochs"],
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0),
            data_dir=data_dir,
        )

    return client_fn


def make_hybrid_client_fn(cfg, data_dir="data/processed"):
    """
    Фабрика клиентов для гибридного сценария.
    """
    def client_fn(cid):
        client_id = int(cid)
        data = load_client_data(client_id, data_dir)

        train_ds = HybridClientDataset(data["train"])
        val_ds = HybridClientDataset(data["val"])

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                                shuffle=False, num_workers=0)

        model = HybridNCF(
            num_public_users=data["num_public_users"],
            num_private_users=data["num_private_users"],
            num_items=data["num_items"],
            emb_dim=cfg["emb_dim"],
            mlp_layers=cfg.get("mlp_layers"),
            dropout=cfg.get("dropout", 0.2),
        )

        return HybridRecClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=cfg["local_epochs"],
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0),
            data_dir=data_dir,
        )

    return client_fn
