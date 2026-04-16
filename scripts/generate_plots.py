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

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('whitegrid')

DATA = Path('data/processed')
RESULTS = Path('data/results')
OUT = Path('report/images')
OUT.mkdir(parents=True, exist_ok=True)


def load_pkl(path):
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_data_distribution(info, data_dir, out_path):
    """Распределение данных по клиентам."""
    rows = []
    priv_genre_matrix = []
    for i in range(info['num_clients']):
        d = load_client_data(i, str(data_dir))
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
    gm_t = gm.T

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.3]})

    axes[0].bar(df_cl['client'], df_cl['priv_users'], color='#FF5722', alpha=0.85)
    axes[0].set_xlabel('Клиент')
    axes[0].set_ylabel('Приватные пользователи')
    axes[0].set_title('Распределение приватных пользователей')
    axes[0].set_xticks(range(1, info['num_clients'] + 1))

    sns.heatmap(gm_t, annot=False, cmap='YlOrRd',
                xticklabels=list(gm_t.columns), yticklabels=list(gm_t.index),
                ax=axes[1])
    axes[1].set_title('Жанровое распределение')
    axes[1].set_xlabel('Клиент')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] {out_path}")


def plot_comparison_curves(ch, fh, out_path):
    """Кривые обучения: centralized vs FL."""
    centr_train_loss = ch['train_loss']
    centr_val_rmse = ch['val_rmse']
    centr_epochs = list(range(1, len(centr_train_loss) + 1))

    fl_fit_loss = [v for _, v in fh.get('metrics_distributed_fit', {}).get('loss', [])]
    fl_fit_rounds = [r for r, _ in fh.get('metrics_distributed_fit', {}).get('loss', [])]
    fl_eval_rmse = [v for _, v in fh.get('metrics_distributed', {}).get('rmse', [])]
    fl_eval_rounds = [r for r, _ in fh.get('metrics_distributed', {}).get('rmse', [])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(centr_epochs, centr_train_loss, '-', color='#2196F3', linewidth=2,
                 label=f'Общее ({len(centr_epochs)} эпох)', alpha=0.9)
    axes[0].plot(fl_fit_rounds, fl_fit_loss, 's-', color='#FF5722', linewidth=2,
                 markersize=5, label=f'FL ({len(fl_fit_rounds)} раундов)', alpha=0.9)
    axes[0].set_xlabel('Шаг обучения')
    axes[0].set_ylabel('Train Loss (MSE)')
    axes[0].set_title('Train Loss')
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].plot(centr_epochs, centr_val_rmse, '-', color='#2196F3', linewidth=2,
                 label='Общее (val)', alpha=0.9)
    axes[1].plot(fl_eval_rounds, fl_eval_rmse, 's-', color='#FF5722', linewidth=2,
                 markersize=5, label='FL (eval)', alpha=0.9)
    axes[1].axhline(ch['test_rmse'], color='#2196F3', linestyle='--', alpha=0.5,
                    label=f'Общее test = {ch["test_rmse"]:.3f}')
    if 'test_rmse' in fh:
        axes[1].axhline(fh['test_rmse'], color='#FF5722', linestyle='--', alpha=0.5,
                        label=f'FL test = {fh["test_rmse"]:.3f}')
    axes[1].set_xlabel('Шаг обучения')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE на валидации')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] {out_path}")


def plot_all_methods_comparison(dataset_name, out_path):
    """Сравнение всех методов (столбчатая диаграмма)."""
    res_dir = RESULTS / dataset_name
    methods = {
        'Centralized': ('centralized', 'centralized_history.pkl'),
        'FedAvg': ('fedavg', 'fl_history.pkl'),
        'FedProx': ('fedprox', 'fl_history.pkl'),
        'FedAvg+DP (σ=0.5)': ('fedavg_dp_s05', 'fl_history.pkl'),
        'FedProx+DP (σ=0.5)': ('fedprox_dp_s05', 'fl_history.pkl'),
        'FedAvg+DP (σ=0.1)': ('fedavg_dp_s01', 'fl_history.pkl'),
        'FedProx+DP (σ=0.1)': ('fedprox_dp_s01', 'fl_history.pkl'),
    }

    metric_names = ['RMSE', 'MAE', 'HR@10', 'NDCG@10']
    results = {}

    for label, (folder, fname) in methods.items():
        data = load_pkl(res_dir / folder / fname)
        if data is None:
            continue

        ranking = data.get('ranking', {})
        results[label] = [
            data.get('test_rmse', ranking.get('rmse', 0)),
            data.get('test_mae', ranking.get('mae', 0)),
            data.get('test_hr10', ranking.get('hr@10', 0)),
            data.get('test_ndcg10', ranking.get('ndcg@10', 0)),
        ]

    if not results:
        print(f"[SKIP] {out_path} — нет данных")
        return

    x = np.arange(len(metric_names))
    n = len(results)
    w = 0.8 / n
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0',
             '#F44336', '#3F51B5']

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (label, vals) in enumerate(results.items()):
        offset = (idx - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=colors[idx % len(colors)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Значение')
    title = 'MovieLens-1M' if 'ml' in dataset_name else 'Amazon Digital Music'
    ax.set_title(f'Сравнение методов: {title}')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] {out_path}")


def plot_per_client(fh, ch, out_path):
    """RMSE и HR@10 по клиентам."""
    if 'per_client_rmse' not in fh:
        print(f"[SKIP] {out_path} — нет per_client_rmse")
        return

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
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] {out_path}")


def main():
    info = load_global_info(str(DATA))
    ch = load_pkl(DATA / 'centralized_history.pkl')
    fh = load_pkl(DATA / 'fl_history.pkl')

    print(f"Данные: {info['num_clients']} клиентов, "
          f"{info['total_users']} юзеров, {info['total_ratings']} рейтингов")

    # Основные графики (из последнего прогона)
    if info and ch:
        plot_data_distribution(info, DATA, OUT / 'data_distribution.png')
    if ch and fh:
        plot_comparison_curves(ch, fh, OUT / 'comparison_curves.png')
        plot_per_client(fh, ch, OUT / 'fl_convergence.png')

    # Сравнение всех методов (если есть результаты)
    for ds in ['ml-1m', 'amazon-music']:
        if (RESULTS / ds).exists():
            plot_all_methods_comparison(ds, OUT / f'comparison_{ds}.png')

    print("\nГотово!")


if __name__ == '__main__':
    main()
