"""Compare seven RF configurations using polygon-grouped cross-validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


P = Path('/mnt/data')
df = pd.read_csv(P / 'Training_Pixels_20m.csv')
mp = pd.read_csv(P / 'ERST619_training_pixel_polygon_mapping.csv')

assert len(df) == len(mp) and np.all(df.pointid.values == mp.pointid.values)

mask = mp.group.notna()
df = df.loc[mask].reset_index(drop=True)
mp = mp.loc[mask].reset_index(drop=True)
y = df.Class_Name
g = mp.group

sets = {
    'RF 4 original': ['dNBR', 'B8', 'NDVI', 'B4'],
    'RF 8 original': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'dNBR'],
    'RF 6 spectral': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
    'RF 6 + SWIR': ['dNBR', 'B8', 'NDVI', 'B4', 'B11', 'B12'],
    'RF 5 + B11': ['dNBR', 'B8', 'NDVI', 'B4', 'B11'],
    'RF 3 indices/NIR': ['dNBR', 'B8', 'NDVI'],
    'RF 4 balanced': ['dNBR', 'B8', 'NDVI', 'B4'],
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(df, y, g))
rows = []
folds = []
preds = {}

for label, ft in sets.items():
    pred = np.empty(len(df), dtype=object)

    for fold, (ti, vi) in enumerate(splits, 1):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if 'balanced' in label else None,
        )
        model.fit(df.loc[ti, ft], y.iloc[ti])
        p = model.predict(df.loc[vi, ft])
        pred[vi] = p

        folds.append({
            'model': label,
            'fold': fold,
            'n_test': len(vi),
            'test_classes': ';'.join(sorted(y.iloc[vi].unique())),
            'accuracy': accuracy_score(y.iloc[vi], p),
            'macroF1': f1_score(y.iloc[vi], p, average='macro', zero_division=0),
        })

    rows.append({
        'model': label,
        'n_features': len(ft),
        'pooled_OA': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_F1': f1_score(y, pred, average='macro', zero_division=0),
        'weighted_F1': f1_score(y, pred, average='weighted', zero_division=0),
        'mean_fold_OA': np.mean([
            r['accuracy'] for r in folds if r['model'] == label
        ]),
    })
    preds[label] = pred

res = pd.DataFrame(rows).sort_values('pooled_OA', ascending=False)
res.to_csv(P / 'ERST619_feature_comparison.csv', index=False)
pd.DataFrame(folds).to_csv(P / 'ERST619_feature_comparison_folds.csv', index=False)

print('MATCHED', len(df), 'GROUPS', g.nunique())
print(res.round(4).to_string(index=False))

for label in ['RF 4 original', res.iloc[0]['model'], 'RF 8 original']:
    print('CONFUSION', label)
    print(confusion_matrix(
        y,
        preds[label],
        labels=['Burned', 'Unburned', 'Man-made', 'Mixed'],
    ))

print('FOLD DETAILS')
print(pd.DataFrame(folds).pivot(
    index='fold', columns='model', values='accuracy'
).round(3).to_string())
print('FOLD TEST CLASSES', [
    (i + 1, sorted(y.iloc[vi].unique().tolist()), len(vi))
    for i, (ti, vi) in enumerate(splits)
])
