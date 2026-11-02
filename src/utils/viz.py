import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def plot_training_curves(metrics_dict, save_path, title='Training Curves', xlabel='Round', ylabel='Value', figsize=(12, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    for (metric_name, values) in metrics_dict.items():
        rounds = list(range(len(values)))
        ax.plot(rounds, values, marker='o', label=metric_name, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_comparison(results_dict, save_path, metric_name='accuracy', title=None, figsize=(12, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = f'{metric_name.capitalize()} Comparison Across Methods'
    (fig, ax) = plt.subplots(figsize=figsize)
    for (method_name, metrics) in results_dict.items():
        if metric_name in metrics:
            values = metrics[metric_name]
            rounds = list(range(len(values)))
            ax.plot(rounds, values, marker='o', label=method_name, linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_client_distribution(client_data, save_path, title='Client Data Distribution', figsize=(14, 10)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    client_ids = sorted(client_data.keys())
    all_labels = set()
    for data in client_data.values():
        all_labels.update(data.keys())
    all_labels = sorted(all_labels)
    num_clients = len(client_ids)
    num_labels = len(all_labels)
    if num_labels == 0:
        return
    matrix = np.zeros((num_clients, num_labels))
    for (i, client_id) in enumerate(client_ids):
        for (j, label) in enumerate(all_labels):
            matrix[i, j] = client_data[client_id].get(label, 0)
    total_per_client = matrix.sum(axis=1)
    total_per_label = matrix.sum(axis=0)
    proportions = matrix / (total_per_client[:, np.newaxis] + 1e-10)
    if num_clients <= 20:
        _plot_client_distribution_bars(matrix, proportions, total_per_client, total_per_label, client_ids, all_labels, num_clients, num_labels, save_path, title, figsize)
    else:
        _plot_client_distribution_heatmap(matrix, proportions, total_per_client, total_per_label, client_ids, all_labels, num_clients, num_labels, save_path, title, figsize)

def _plot_client_distribution_bars(matrix, proportions, total_per_client, total_per_label, client_ids, all_labels, num_clients, num_labels, save_path, title, figsize):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1], hspace=0.3, wspace=0.15)
    if num_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_labels))
    else:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_labels))
    ax_main = fig.add_subplot(gs[0, 0])
    bar_positions = np.arange(num_clients)
    bar_height = 0.7
    left = np.zeros(num_clients)
    for (j, label) in enumerate(all_labels):
        ax_main.barh(bar_positions, proportions[:, j], left=left, height=bar_height, label=f'Class {label}', color=colors[j], edgecolor='white', linewidth=0.5)
        left += proportions[:, j]
    ax_main.set_yticks(bar_positions)
    ax_main.set_yticklabels([f'Client {c}' for c in client_ids], fontsize=9)
    ax_main.set_xlabel('Label Proportion', fontsize=11)
    ax_main.set_xlim(0, 1)
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title='Labels', title_fontsize=9, framealpha=0.9)
    ax_main.set_axisbelow(True)
    ax_main.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax_main.invert_yaxis()
    ax_counts = fig.add_subplot(gs[0, 1])
    bar_colors = plt.cm.Blues(0.3 + 0.5 * total_per_client / total_per_client.max())
    ax_counts.barh(bar_positions, total_per_client, height=bar_height, color=bar_colors, edgecolor='darkblue', linewidth=0.5)
    ax_counts.set_yticks([])
    ax_counts.set_xlabel('Samples', fontsize=10)
    ax_counts.set_title('Total', fontsize=11, fontweight='bold')
    ax_counts.invert_yaxis()
    for (i, count) in enumerate(total_per_client):
        ax_counts.text(count + total_per_client.max() * 0.02, i, f'{int(count)}', va='center', fontsize=7)
    ax_counts.set_xlim(0, total_per_client.max() * 1.15)
    ax_counts.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax_counts.set_axisbelow(True)
    ax_summary = fig.add_subplot(gs[1, 0])
    x_positions = np.arange(num_labels)
    bars = ax_summary.bar(x_positions, total_per_label, color=colors, edgecolor='white', linewidth=0.5)
    ax_summary.set_xticks(x_positions)
    ax_summary.set_xticklabels([f'Class {lbl}' for lbl in all_labels], fontsize=9)
    ax_summary.set_ylabel('Total Samples', fontsize=10)
    ax_summary.set_title('Overall Class Distribution', fontsize=11, fontweight='bold')
    for (bar, val) in zip(bars, total_per_label):
        ax_summary.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(val)}', ha='center', va='bottom', fontsize=8)
    ax_summary.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax_summary.set_axisbelow(True)
    _add_stats_panel(fig.add_subplot(gs[1, 1]), matrix, total_per_client, total_per_label, num_clients, num_labels)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def _plot_client_distribution_heatmap(matrix, proportions, total_per_client, total_per_label, client_ids, all_labels, num_clients, num_labels, save_path, title, figsize):
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.15, 2.5, 1], width_ratios=[2.5, 0.8, 1.2], hspace=0.25, wspace=0.2)
    if num_labels <= 10:
        class_colors = plt.cm.tab10(np.linspace(0, 1, num_labels))
    else:
        class_colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_labels))
    ax_legend = fig.add_subplot(gs[0, 0])
    for (j, label) in enumerate(all_labels):
        ax_legend.barh(0, 1, left=j, height=0.6, color=class_colors[j], edgecolor='white', linewidth=1)
        ax_legend.text(j + 0.5, 0, f'C{label}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax_legend.set_xlim(0, num_labels)
    ax_legend.set_ylim(-0.5, 0.5)
    ax_legend.axis('off')
    ax_legend.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax_heatmap = fig.add_subplot(gs[1, 0])
    dominant_class = np.argmax(proportions, axis=1)
    sort_idx = np.lexsort((total_per_client, dominant_class))
    proportions_sorted = proportions[sort_idx]
    total_sorted = total_per_client[sort_idx]
    im = ax_heatmap.imshow(proportions_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax_heatmap, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Proportion', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax_heatmap.set_xlabel('Class', fontsize=11)
    ax_heatmap.set_ylabel(f'Clients (sorted by dominant class, n={num_clients})', fontsize=11)
    ax_heatmap.set_xticks(np.arange(num_labels))
    ax_heatmap.set_xticklabels([f'{lbl}' for lbl in all_labels], fontsize=9)
    if num_clients > 50:
        ytick_step = num_clients // 10
        yticks = np.arange(0, num_clients, ytick_step)
        ax_heatmap.set_yticks(yticks)
        ax_heatmap.set_yticklabels([f'{i}' for i in yticks], fontsize=8)
    else:
        ax_heatmap.set_yticks(np.arange(num_clients))
        ax_heatmap.set_yticklabels(np.arange(num_clients), fontsize=7)
    ax_samples = fig.add_subplot(gs[1, 1])
    y_pos = np.arange(num_clients)
    colors_by_size = plt.cm.Blues(0.3 + 0.6 * total_sorted / total_sorted.max())
    ax_samples.barh(y_pos, total_sorted, color=colors_by_size, edgecolor='none', height=1.0)
    ax_samples.set_ylim(-0.5, num_clients - 0.5)
    ax_samples.set_xlim(0, total_sorted.max() * 1.1)
    ax_samples.set_xlabel('Samples', fontsize=10)
    ax_samples.set_title('Per Client', fontsize=10, fontweight='bold')
    ax_samples.set_yticks([])
    ax_samples.invert_yaxis()
    ax_samples.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax_samples.axvline(total_per_client.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {total_per_client.mean():.0f}')
    ax_samples.legend(fontsize=8, loc='lower right')
    ax_hist = fig.add_subplot(gs[2, 0])
    bins = min(30, num_clients // 3)
    (n, bins_edges, patches) = ax_hist.hist(total_per_client, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
    ax_hist.axvline(total_per_client.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {total_per_client.mean():.0f}')
    ax_hist.axvline(np.median(total_per_client), color='orange', linestyle='-.', linewidth=2, label=f'Median: {np.median(total_per_client):.0f}')
    ax_hist.set_xlabel('Samples per Client', fontsize=11)
    ax_hist.set_ylabel('Number of Clients', fontsize=11)
    ax_hist.set_title('Client Size Distribution', fontsize=11, fontweight='bold')
    ax_hist.legend(fontsize=9)
    ax_hist.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_class = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(num_labels)
    bars = ax_class.bar(x_pos, total_per_label, color=class_colors, edgecolor='white', linewidth=0.5)
    ax_class.set_xticks(x_pos)
    ax_class.set_xticklabels([f'{lbl}' for lbl in all_labels], fontsize=9)
    ax_class.set_ylabel('Samples', fontsize=10)
    ax_class.set_title('Class Totals', fontsize=10, fontweight='bold')
    ax_class.yaxis.grid(True, linestyle='--', alpha=0.3)
    total = total_per_label.sum()
    for (bar, val) in zip(bars, total_per_label):
        pct = 100 * val / total
        ax_class.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax_stats = fig.add_subplot(gs[2, 2])
    _add_stats_panel_extended(ax_stats, matrix, proportions, total_per_client, total_per_label, num_clients, num_labels)
    ax_hetero = fig.add_subplot(gs[0, 1:])
    ax_hetero.axis('off')
    uniform = np.ones(num_labels) / num_labels
    kl_divs = []
    for i in range(num_clients):
        p = proportions[i] + 1e-10
        kl = np.sum(p * np.log(p / uniform))
        kl_divs.append(kl)
    avg_kl = np.mean(kl_divs)
    if avg_kl < 0.1:
        (hetero_level, hetero_color) = ('LOW (IID-like)', 'green')
    elif avg_kl < 0.5:
        (hetero_level, hetero_color) = ('MODERATE', 'orange')
    else:
        (hetero_level, hetero_color) = ('HIGH (Non-IID)', 'red')
    ax_hetero.text(0.5, 0.5, f'Data Heterogeneity: {hetero_level}', transform=ax_hetero.transAxes, fontsize=12, fontweight='bold', ha='center', va='center', color=hetero_color, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=hetero_color, linewidth=2))
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def _add_stats_panel(ax, matrix, total_per_client, total_per_label, num_clients, num_labels):
    ax.axis('off')
    total_samples = matrix.sum()
    avg_per_client = total_per_client.mean()
    std_per_client = total_per_client.std()
    min_samples = total_per_client.min()
    max_samples = total_per_client.max()
    imbalance_ratio = total_per_label.max() / (total_per_label.min() + 1e-10)
    stats_text = f"Summary Statistics\n{'─' * 20}\nTotal Samples: {int(total_samples):,}\nClients: {num_clients}\nClasses: {num_labels}\n{'─' * 20}\nAvg/Client: {avg_per_client:.0f}\nStd/Client: {std_per_client:.0f}\nMin/Max: {int(min_samples)}/{int(max_samples)}\n{'─' * 20}\nClass Imbalance: {imbalance_ratio:.2f}x"
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='center', horizontalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='gray', alpha=0.8))

def _add_stats_panel_extended(ax, matrix, proportions, total_per_client, total_per_label, num_clients, num_labels):
    ax.axis('off')
    total_samples = matrix.sum()
    avg_per_client = total_per_client.mean()
    std_per_client = total_per_client.std()
    cv = std_per_client / (avg_per_client + 1e-10)
    sorted_sizes = np.sort(total_per_client)
    n = len(sorted_sizes)
    cumsum = np.cumsum(sorted_sizes)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_sizes) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
    imbalance_ratio = total_per_label.max() / (total_per_label.min() + 1e-10)
    stats_text = f"  Summary Statistics\n  {'─' * 22}\n  Total: {int(total_samples):,} samples\n  Clients: {num_clients}\n  Classes: {num_labels}\n  {'─' * 22}\n  Client Size Stats:\n    Mean: {avg_per_client:,.0f}\n    Std:  {std_per_client:,.0f}\n    CV:   {cv:.2f}\n    Gini: {gini:.3f}\n  {'─' * 22}\n  Class Imbalance: {imbalance_ratio:.2f}x"
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='center', horizontalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#888888', alpha=0.95))

def plot_per_domain_acc(domain_metrics, save_path, title='Per-Domain Accuracy (DomainNet)', figsize=(12, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    for (domain_name, accuracies) in domain_metrics.items():
        rounds = list(range(len(accuracies)))
        ax.plot(rounds, accuracies, marker='o', label=domain_name.capitalize(), linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_communication_cost(comm_metrics, save_path, title='Communication Cost Over Rounds', figsize=(12, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    for (metric_name, values) in comm_metrics.items():
        values_mb = [v / 1024 ** 2 for v in values]
        rounds = list(range(len(values)))
        ax.plot(rounds, values_mb, marker='o', label=metric_name.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Communication (MB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_and_accuracy(metrics, save_path, title='Training Progress', figsize=(14, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)
    loss_key = None
    for key in metrics.keys():
        if 'loss' in key.lower():
            loss_key = key
            break
    if loss_key:
        rounds = list(range(len(metrics[loss_key])))
        ax1.plot(rounds, metrics[loss_key], marker='o', color='red', linewidth=2)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    acc_key = None
    for key in metrics.keys():
        if 'acc' in key.lower():
            acc_key = key
            break
    if acc_key:
        rounds = list(range(len(metrics[acc_key])))
        ax2.plot(rounds, metrics[acc_key], marker='o', color='green', linewidth=2)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Test Accuracy', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_comparison(methods, final_scores, save_path, metric_name='Accuracy', title=None, figsize=(10, 6)):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = f'Final {metric_name} Comparison'
    (fig, ax) = plt.subplots(figsize=figsize)
    colors = sns.color_palette('husl', len(methods))
    bars = ax.bar(methods, final_scores, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(final_scores) * 1.1)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_comparison(results_dict, save_path, metric_name='accuracy', title='Convergence Comparison', figsize=(12, 6), smoothing=False, window_size=3):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=figsize)
    for (method_name, values) in results_dict.items():
        rounds = list(range(len(values)))
        if smoothing and len(values) > window_size:
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
            rounds_smoothed = rounds[window_size - 1:]
            ax.plot(rounds, values, alpha=0.3, linewidth=1)
            ax.plot(rounds_smoothed, smoothed, marker='o', label=method_name, linewidth=2)
        else:
            ax.plot(rounds, values, marker='o', label=method_name, linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_summary(results_dict, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

def load_metrics_from_json(json_path):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f'Metrics file not found: {json_path}')
    with open(json_path, 'r') as f:
        return json.load(f)
