"""Генерация графиков для отчёта из сохранённых данных (без повторного обучения)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.splitter import load_global_info, load_client_data

# Настройки matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

DATA = Path('data/processed')
OUT = Path('report/images')
OUT.mkdir(parents=True, exist_ok=True)

# --- Загрузка данных ---
info = load_global_info(str(DATA))

with open(DATA / 'centralized_history.pkl', 'rb') as f:
    ch = pickle.load(f)

with open(DATA / 'fl_history.pkl', 'rb') as f:
    fh = pickle.load(f)

print(f"Данные загружены: {info['num_clients']} клиентов, "
      f"{info['total_users']} юзеров, {info['total_ratings']} рейтингов")


rows = []
priv_genre_matrix = []
for i in range(info['num_clients']):
    d = load_client_data(i, str(DATA))
    n_priv = d.get('num_private_users', d.get('num_users', 0))
    rows.append({'client': i + 1, 'priv_users': n_priv})
    if 'is_public' in d['train'].columns:
        priv_train = d['train'][d['train']['is_public'] == 0]
        if 'genre' in priv_train.columns and len(priv_train) > 0:
            gd = priv_train['genre'].value_counts(normalize=True).to_dict()
        else:
            gd = d['genre_distribution']
    else:
        gd = d['genre_distribution']
    priv_genre_matrix.append(gd)

df_cl = pd.DataFrame(rows)

gm = pd.DataFrame(priv_genre_matrix).fillna(0)
gm.index = [f'{i+1}' for i in range(len(gm))]
gm_t = gm.T  # жанры по Y, клиенты по X

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.3]})

# Левый: столбцы приватных юзеров
axes[0].bar(df_cl['client'], df_cl['priv_users'], color='#FF5722', alpha=0.85)
axes[0].set_xlabel('Клиент')
axes[0].set_ylabel('Приватные пользователи')
axes[0].set_title('Распределение приватных пользователей')
axes[0].set_xticks(range(1, info['num_clients'] + 1))

# Правый: жанровая heatmap (клиенты по X, жанры по Y)
sns.heatmap(gm_t, annot=False, cmap='YlOrRd',
            xticklabels=list(gm_t.columns), yticklabels=list(gm_t.index),
            ax=axes[1])
axes[1].set_title('Жанровое распределение')
axes[1].set_xlabel('Клиент')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(OUT / 'data_distribution.png')
plt.close()
print(f"[OK] {OUT / 'data_distribution.png'}")


centr_train_loss = ch['train_loss']
centr_val_rmse = ch['val_rmse']
centr_epochs = list(range(1, len(centr_train_loss) + 1))

fl_fit_loss = [v for _, v in fh.get('metrics_distributed_fit', {}).get('loss', [])]
fl_fit_rounds = [r for r, _ in fh.get('metrics_distributed_fit', {}).get('loss', [])]
fl_eval_rmse = [v for _, v in fh.get('metrics_distributed', {}).get('rmse', [])]
fl_eval_rounds = [r for r, _ in fh.get('metrics_distributed', {}).get('rmse', [])]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
ax1.plot(centr_epochs, centr_train_loss, '-', color='#2196F3', linewidth=2,
         label=f'Общее обучение ({len(centr_epochs)} эпох)', alpha=0.9)
ax1.plot(fl_fit_rounds, fl_fit_loss, 's-', color='#FF5722', linewidth=2,
         markersize=5, label=f'Федеративное ({len(fl_fit_rounds)} раундов)', alpha=0.9)
ax1.set_xlabel('Шаг обучения (эпоха / раунд)')
ax1.set_ylabel('Train Loss (MSE)')
ax1.set_title('Сравнение: Train Loss')
ax1.legend()
ax1.set_ylim(bottom=0)

ax2 = axes[1]
ax2.plot(centr_epochs, centr_val_rmse, '-', color='#2196F3', linewidth=2,
         label=f'Общее (val)', alpha=0.9)
ax2.plot(fl_eval_rounds, fl_eval_rmse, 's-', color='#FF5722', linewidth=2,
         markersize=5, label=f'Федеративное (eval)', alpha=0.9)
ax2.axhline(ch['test_rmse'], color='#2196F3', linestyle='--', alpha=0.5,
            label=f'Общее test RMSE = {ch["test_rmse"]:.3f}')
if 'test_rmse' in fh:
    ax2.axhline(fh['test_rmse'], color='#FF5722', linestyle='--', alpha=0.5,
                label=f'FL test RMSE = {fh["test_rmse"]:.3f}')
ax2.set_xlabel('Шаг обучения (эпоха / раунд)')
ax2.set_ylabel('RMSE')
ax2.set_title('Сравнение: RMSE на валидации')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'comparison_curves.png')
plt.close()
print(f"[OK] {OUT / 'comparison_curves.png'}")


centr_ranking = ch.get('ranking', {})
metrics_names = ['RMSE', 'MAE', 'HR@10', 'NDCG@10']
centr_vals = [
    ch['test_rmse'], ch['test_mae'],
    centr_ranking.get('hr@10', 0), centr_ranking.get('ndcg@10', 0)
]
fl_vals = [
    fh.get('test_rmse', 0), fh.get('test_mae', 0),
    fh.get('test_hr10', 0), fh.get('test_ndcg10', 0)
]

x = np.arange(len(metrics_names))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - w/2, centr_vals, w, label='Общее обучение', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + w/2, fl_vals, w, label='Федеративное', color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylabel('Значение')
ax.set_title('Сравнение метрик: Общее vs Федеративное обучение')
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'comparison_metrics.png')
plt.close()
print(f"[OK] {OUT / 'comparison_metrics.png'}")


if 'per_client_rmse' in fh:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_cl = len(fh['per_client_rmse'])
    clients = list(range(1, n_cl + 1))
    colors = sns.color_palette('viridis', n_cl)

    axes[0].bar(clients, fh['per_client_rmse'], color=colors)
    axes[0].axhline(fh['test_rmse'], color='red', linestyle='--', alpha=0.7,
                    label=f'Среднее = {fh["test_rmse"]:.3f}')
    axes[0].axhline(ch['test_rmse'], color='blue', linestyle='--', alpha=0.7,
                    label=f'Общее = {ch["test_rmse"]:.3f}')
    axes[0].set_xlabel('Клиент')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('FL: RMSE по клиентам')
    axes[0].set_xticks(clients)
    axes[0].legend(fontsize=9)
    rmse_vals = fh['per_client_rmse']
    pad = 0.02
    axes[0].set_ylim(min(rmse_vals) - pad, max(max(rmse_vals), ch['test_rmse']) + pad)

    if 'per_client_hr10' in fh:
        axes[1].bar(clients, fh['per_client_hr10'], color=colors)
        axes[1].axhline(fh['test_hr10'], color='red', linestyle='--', alpha=0.7,
                        label=f'Среднее = {fh["test_hr10"]:.3f}')
        centr_hr10 = ch.get('ranking', {}).get('hr@10', 0)
        axes[1].axhline(centr_hr10, color='blue', linestyle='--', alpha=0.7,
                        label=f'Общее = {centr_hr10:.3f}')
        axes[1].set_xlabel('Клиент')
        axes[1].set_ylabel('HR@10')
        axes[1].set_title('FL: Hit Rate@10 по клиентам')
        axes[1].set_xticks(clients)
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'fl_convergence.png')
    plt.close()
    print(f"[OK] {OUT / 'fl_convergence.png'}")
else:
    print("[SKIP] fl_convergence.png — нет per_client_rmse в данных")

print("\nГотово!")
