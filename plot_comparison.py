import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'))

COLOR_CORR = '#008300'   # green
COLOR_MAE = '#2a78d6'    # blue
COLOR_RMSE = '#eb6834'   # orange
GRID = '#e1e0d9'
TICK = '#898781'
LABEL = '#52514e'
INK = '#0b0b0b'

models = df['Model'].tolist()
x = np.arange(len(models))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: Correlation (unitless, 0-1)
axes[0].bar(x, df['Correlation'], width=0.55, color=COLOR_CORR, zorder=3)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=40, ha='right', color=INK)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Correlation', color=INK)
axes[0].set_title('Correlation (higher is better)', color=INK, fontsize=12)
axes[0].grid(axis='y', color=GRID, linewidth=1, zorder=0)
axes[0].set_axisbelow(True)
axes[0].tick_params(colors=TICK)
for spine in ['top', 'right']:
    axes[0].spines[spine].set_visible(False)
for i, v in enumerate(df['Correlation']):
    axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, color=LABEL)

# Panel 2: MAE / RMSE share units (bikes), same axis is appropriate here
width = 0.35
axes[1].bar(x - width / 2, df['MAE'], width, label='MAE', color=COLOR_MAE, zorder=3)
axes[1].bar(x + width / 2, df['RMSE'], width, label='RMSE', color=COLOR_RMSE, zorder=3)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=40, ha='right', color=INK)
axes[1].set_ylabel('Bikes', color=INK)
axes[1].set_title('Error (lower is better)', color=INK, fontsize=12)
axes[1].grid(axis='y', color=GRID, linewidth=1, zorder=0)
axes[1].set_axisbelow(True)
axes[1].tick_params(colors=TICK)
for spine in ['top', 'right']:
    axes[1].spines[spine].set_visible(False)
axes[1].legend(frameon=False, labelcolor=INK)

fig.suptitle('Model Performance Comparison', fontsize=14, color=INK)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=300)
print(f'Saved: {FIGURES_DIR}/model_comparison.png')
