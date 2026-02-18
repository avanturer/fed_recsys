"""NCF — Neural Collaborative Filtering (He et al., 2017)."""
import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Две ветки: GMF (обобщённая матричная факторизация) и MLP.
    GMF ловит линейные паттерны user-item, MLP — нелинейные.
    На выходе — предсказанный рейтинг [1, 5].
    """

    def __init__(self, num_users, num_items, emb_dim=64,
                 mlp_layers=None, dropout=0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items

        # GMF-эмбеддинги
        self.gmf_user_emb = nn.Embedding(num_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP-эмбеддинги (отдельные, чтобы GMF и MLP учились независимо)
        self.mlp_user_emb = nn.Embedding(num_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP
        layers = []
        in_dim = emb_dim * 2
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        # Финальный слой: конкатенация GMF-вектора и MLP-выхода -> рейтинг
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                     self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF: поэлементное произведение эмбеддингов
        gmf = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)

        # MLP: конкатенация -> нелинейные слои
        mlp_in = torch.cat([
            self.mlp_user_emb(user_ids),
            self.mlp_item_emb(item_ids)
        ], dim=-1)
        mlp_out = self.mlp(mlp_in)

        out = self.output(torch.cat([gmf, mlp_out], dim=-1))
        return (torch.sigmoid(out) * 4 + 1).squeeze(-1)

    def shared_param_keys(self):
        """Ключи параметров для федеративной агрегации — всё кроме эмбеддингов.

        Эмбеддинги у каждого клиента свои (разные юзеры/айтемы),
        а MLP и output-слой учат общие паттерны взаимодействия —
        их и агрегируем через FedAvg.
        """
        return [k for k in self.state_dict().keys() if 'emb' not in k]

    def get_shared_params(self):
        """Достаём shared-параметры как list[np.ndarray] для отправки на сервер."""
        state = self.state_dict()
        return [state[k].cpu().numpy() for k in self.shared_param_keys()]

    def load_shared_params(self, params_list):
        """Загружаем shared-параметры с сервера, эмбеддинги не трогаем."""
        state = self.state_dict()
        for key, val in zip(self.shared_param_keys(), params_list):
            state[key] = torch.as_tensor(val)
        self.load_state_dict(state)


class HybridNCF(nn.Module):
    """
    NCF для гибридного сценария: публичные + приватные пользователи.

    Публичные юзеры — глобальные эмбеддинги (агрегируются).
    Приватные юзеры — локальные эмбеддинги (остаются на клиенте).
    Айтемы — глобальные эмбеддинги (агрегируются).

    Forward принимает:
    - user_ids: tensor с ID пользователей
    - item_ids: tensor с ID айтемов
    - is_public: tensor[bool] с флагами (True=публичный, False=приватный)
    """

    def __init__(self, num_public_users, num_private_users, num_items,
                 emb_dim=64, mlp_layers=None, dropout=0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.num_public_users = num_public_users
        self.num_private_users = num_private_users
        self.num_items = num_items

        # GMF: публичные + приватные + айтемы
        self.gmf_public_user_emb = nn.Embedding(num_public_users, emb_dim)
        self.gmf_private_user_emb = nn.Embedding(num_private_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP: публичные + приватные + айтемы
        self.mlp_public_user_emb = nn.Embedding(num_public_users, emb_dim)
        self.mlp_private_user_emb = nn.Embedding(num_private_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP слои
        layers = []
        in_dim = emb_dim * 2
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        # Output
        self.output = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_public_user_emb, self.gmf_private_user_emb,
                    self.gmf_item_emb, self.mlp_public_user_emb,
                    self.mlp_private_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids, is_public):
        """
        Args:
            user_ids: [batch_size] — ID юзера (public или private в зависимости от is_public)
            item_ids: [batch_size] — ID айтема (глобальные)
            is_public: [batch_size] — bool tensor, True если юзер публичный
        """
        batch_size = user_ids.size(0)
        emb_dim = self.gmf_public_user_emb.embedding_dim

        # GMF user embedding: выбираем public или private в зависимости от is_public
        gmf_user = torch.zeros(batch_size, emb_dim, device=user_ids.device)
        if is_public.any():
            pub_mask = is_public
            gmf_user[pub_mask] = self.gmf_public_user_emb(user_ids[pub_mask])
        if (~is_public).any():
            priv_mask = ~is_public
            gmf_user[priv_mask] = self.gmf_private_user_emb(user_ids[priv_mask])

        # MLP user embedding
        mlp_user = torch.zeros(batch_size, emb_dim, device=user_ids.device)
        if is_public.any():
            pub_mask = is_public
            mlp_user[pub_mask] = self.mlp_public_user_emb(user_ids[pub_mask])
        if (~is_public).any():
            priv_mask = ~is_public
            mlp_user[priv_mask] = self.mlp_private_user_emb(user_ids[priv_mask])

        # Эмбеддинги айтемов
        gmf_item = self.gmf_item_emb(item_ids)
        mlp_item = self.mlp_item_emb(item_ids)

        # GMF: поэлементное произведение
        gmf = gmf_user * gmf_item

        # MLP: конкатенация -> нелинейные слои
        mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Выход
        out = self.output(torch.cat([gmf, mlp_out], dim=-1))
        return (torch.sigmoid(out) * 4 + 1).squeeze(-1)

    def shared_param_keys(self):
        """
        Shared-параметры для агрегации:
        - Публичные user embeddings (все клиенты видят одних и тех же публичных юзеров)
        - Item embeddings (все айтемы глобальные)
        - MLP + output (общие паттерны)

        НЕ агрегируем:
        - Приватные user embeddings (у каждого клиента свои приватные юзеры)
        """
        all_keys = list(self.state_dict().keys())
        shared = [k for k in all_keys if 'private_user_emb' not in k]
        return shared

    def get_shared_params(self):
        """Достаём shared-параметры как list[np.ndarray] для отправки на сервер."""
        state = self.state_dict()
        return [state[k].cpu().numpy() for k in self.shared_param_keys()]

    def load_shared_params(self, params_list):
        """Загружаем shared-параметры с сервера, приватные эмбеддинги не трогаем."""
        state = self.state_dict()
        for key, val in zip(self.shared_param_keys(), params_list):
            state[key] = torch.as_tensor(val)
        self.load_state_dict(state)
