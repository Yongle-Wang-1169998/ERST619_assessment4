"""Generate Pearson correlation, JM separability, and RF importance outputs."""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


out = Path('/mnt/data/ERST619_Final_Figures')
df = pd.read_csv('/mnt/data/Training_Pixels_20m.csv')

features = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'dNBR']
classes = ['Burned', 'Unburned', 'Man-made', 'Mixed']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'savefig.dpi': 300,
})

# Pearson correlation heatmap
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(8.4, 7.2))
im = ax.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticks(range(8), features)
ax.set_yticks(range(8), features)
ax.tick_params(length=0)

for i in range(8):
    for j in range(8):
        r = corr.iat[i, j]
        ax.text(
            j, i, f'{r:.2f}',
            ha='center', va='center', fontsize=9,
            color='white' if abs(r) > 0.65 else 'black',
        )

fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson correlation coefficient (r)')
fig.tight_layout()
fig.savefig(out / 'Figure_3_Pearson_Correlation_Heatmap.png', bbox_inches='tight')
fig.savefig(out / 'Figure_3_Pearson_Correlation_Heatmap.pdf', bbox_inches='tight')
plt.close(fig)

# Jeffries–Matusita separability
rows = []
feature_sets = [
    (features, '8 predictors'),
    (['dNBR', 'B8', 'NDVI', 'B4'], '4 predictors'),
]

for ff, tag in feature_sets:
    a = df[ff].to_numpy(float)
    a = (a - a.mean(axis=0)) / a.std(axis=0)
    labels = df.Class_Name.to_numpy()

    for c1, c2 in combinations(classes, 2):
        v1 = a[labels == c1]
        v2 = a[labels == c2]
        delta = v1.mean(0) - v2.mean(0)

        cov1 = np.cov(v1, rowvar=False) + np.eye(len(ff)) * 1e-6
        cov2 = np.cov(v2, rowvar=False) + np.eye(len(ff)) * 1e-6
        cov = (cov1 + cov2) / 2

        _, d1 = np.linalg.slogdet(cov1)
        _, d2 = np.linalg.slogdet(cov2)
        _, d = np.linalg.slogdet(cov)

        bh = (
            0.125 * delta @ np.linalg.solve(cov, delta)
            + 0.5 * (d - 0.5 * (d1 + d2))
        )
        rows.append({
            'Class pair': f'{c1} – {c2}',
            'Feature set': tag,
            'JM': 2 * (1 - np.exp(-bh)),
        })

tbl = (
    pd.DataFrame(rows)
    .pivot(index='Class pair', columns='Feature set', values='JM')
    .reset_index()[['Class pair', '8 predictors', '4 predictors']]
)
tbl.to_csv(out / 'Table_2_JM_Separability.csv', index=False)

fig, ax = plt.subplots(figsize=(9.0, 3.6))
ax.axis('off')
vals = [
    [r['Class pair'], f"{r['8 predictors']:.3f}", f"{r['4 predictors']:.3f}"]
    for _, r in tbl.iterrows()
]
t = ax.table(
    cellText=vals,
    colLabels=['Class pair', '8 predictors', '4 predictors'],
    cellLoc='center',
    colLoc='center',
    loc='center',
    colWidths=[0.53, 0.235, 0.235],
)
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1, 1.65)

for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor('#cccccc')
    cell.set_linewidth(0.5)
    if i == 0:
        cell.set_facecolor('#e9edf1')
        cell.set_text_props(weight='bold')
    if j == 0 and i > 0:
        cell.get_text().set_ha('left')

fig.tight_layout()
fig.savefig(out / 'Table_2_JM_Separability.png', bbox_inches='tight')
fig.savefig(out / 'Table_2_JM_Separability.pdf', bbox_inches='tight')
plt.close(fig)

# Random Forest feature importance
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(df[features], df['Class_Name'])
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh(imp.index, imp.values, color='#537eaa')
ax.set_xlim(0, 0.26)
ax.set_xlabel('Mean decrease in impurity (feature importance)')

for i, v in enumerate(imp):
    ax.text(v + 0.004, i, f'{v:.3f}', va='center', fontsize=10)

ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out / 'Figure_4_RF_Feature_Importance.png', bbox_inches='tight')
fig.savefig(out / 'Figure_4_RF_Feature_Importance.pdf', bbox_inches='tight')
plt.close(fig)

imp.sort_values(ascending=False).rename('Importance').to_csv(
    out / 'Figure_4_Feature_Importance_Values.csv'
)

print(tbl.round(3).to_string(index=False))
print('\nRF:', imp.sort_values(ascending=False).round(4).to_dict())
print('\nFiles:')
print('\n'.join(str(p) for p in out.iterdir()))
