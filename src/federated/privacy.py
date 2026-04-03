"""Дифференциальная приватность для федеративного обучения."""
import math

import numpy as np


def clip_and_add_noise(local_params, global_params, max_grad_norm, noise_multiplier):
    """
    DP-механизм для обновлений модели перед отправкой на сервер.

    1. delta = local - global
    2. Клиппинг: delta *= min(1, C / ||delta||_2)
    3. Шум: delta += N(0, (sigma * C)^2)
    4. Возврат: global + зашумлённый delta
    """
    deltas = [loc - glob for loc, glob in zip(local_params, global_params)]

    # L2-норма всего вектора обновлений
    flat = np.concatenate([d.ravel() for d in deltas])
    update_norm = np.linalg.norm(flat)

    # Клиппинг
    clip_factor = min(1.0, max_grad_norm / max(update_norm, 1e-10))
    clipped = [d * clip_factor for d in deltas]

    # Гауссов шум
    noise_std = noise_multiplier * max_grad_norm
    noised = [
        d + np.random.normal(0, noise_std, d.shape).astype(d.dtype)
        for d in clipped
    ]

    return [glob + nd for glob, nd in zip(global_params, noised)]


class PrivacyAccountant:
    """Учёт приватности через Gaussian mechanism + композицию."""

    def __init__(self, noise_multiplier, target_delta=1e-5):
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_epsilon(self):
        """epsilon через simple composition."""
        if self.steps == 0 or self.noise_multiplier <= 0:
            return float("inf")
        eps_1 = math.sqrt(2 * math.log(1.25 / self.target_delta)) / self.noise_multiplier
        return self.steps * eps_1

    def get_epsilon_advanced(self):
        """epsilon через advanced composition (более tight bound)."""
        if self.steps == 0 or self.noise_multiplier <= 0:
            return float("inf")
        eps_1 = math.sqrt(2 * math.log(1.25 / self.target_delta)) / self.noise_multiplier
        delta_prime = self.target_delta / 2
        T = self.steps
        eps_adv = (math.sqrt(2 * T * math.log(1 / delta_prime)) * eps_1
                   + T * eps_1 * (math.exp(eps_1) - 1))
        return min(eps_adv, self.get_epsilon())

    def __repr__(self):
        eps = self.get_epsilon()
        return f"eps={eps:.2f}, delta={self.target_delta}, steps={self.steps}"
