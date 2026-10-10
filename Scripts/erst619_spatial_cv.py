"""Map training/validation pixels to polygons and evaluate spatial CV."""

from collections import defaultdict, Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, cross_val_predict


P = Path('/mnt/data')
cols = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
feats = ['dNBR', 'B8', 'NDVI', 'B4']
full = cols + ['NDVI', 'dNBR']

tr = pd.read_csv(P / 'Training_Pixels_20m.csv')
val = pd.read_csv(P / 'Validation_Pixels_20m.csv')
shp = gpd.read_file(P / 'training_shp/Training_20240220.shp')
shpval = gpd.read_file(P / 'validation_shp/Validation_20240220.shp')

with rasterio.open(P / 'Port_Hills_Classification_8Bands_2024.tif') as src:
    arr = src.read()
    tf = src.transform
    loc = defaultdict(list)

    for r, c in zip(*np.where(np.isfinite(arr).all(axis=0))):
        k = tuple(int(arr[i, r, c]) for i in range(6))
        loc[k].append((r, c))

    def assign(df, shapes):
        records = []

        for _, row in df.iterrows():
            k = tuple(int(row[q]) for q in cols)
            candidates = loc.get(k, [])
            candidates = [
                (r, c)
                for r, c in candidates
                if abs(float(arr[6, r, c]) - row.NDVI) < 0.00001
                and abs(float(arr[7, r, c]) - row.dNBR) < 0.00001
            ]
            candidates0 = candidates
            choices = []

            for r, c in candidates:
                x, y = rasterio.transform.xy(tf, r, c)
                p = Point(x, y)
                ids = [
                    i
                    for i, polygon in enumerate(shapes.geometry)
                    if polygon.covers(p)
                ]
                for i in ids:
                    if row.Class_Name == shapes.iloc[i].Class_Name:
                        choices.append((r, c, i))

            records.append({
                'pointid': row.pointid,
                'Class_Name': row.Class_Name,
                'group': shapes.iloc[choices[0][2]].Sample_ID if choices else None,
                'Notes': shapes.iloc[choices[0][2]].Notes if choices else None,
                'possible': len(candidates0),
                'matched': len(choices),
                'r': choices[0][0] if choices else None,
                'c': choices[0][1] if choices else None,
            })

        return pd.DataFrame(records)

    t = assign(tr, shp)
    v = assign(val, shpval)

    print(
        'MATCH train', t.group.notna().sum(), '/', len(t),
        'unique', t.group.nunique(),
        'missing', len(t) - t.group.notna().sum(),
        'ambiguous', sum(t.matched > 1),
    )
    print(
        'MATCH val', v.group.notna().sum(), '/', len(v),
        'unique', v.group.nunique(),
        'missing', len(v) - v.group.notna().sum(),
        'ambiguous', sum(v.matched > 1),
    )
    print('train unmatched by class', t[t.group.isna()].Class_Name.value_counts().to_dict())
    print('train group counts', t.groupby(
        ['Class_Name', 'group', 'Notes']
    ).size().sort_values(ascending=False).head(15).to_string())

    t.to_csv(P / 'ERST619_training_pixel_polygon_mapping.csv', index=False)
    v.to_csv(P / 'ERST619_validation_pixel_polygon_mapping.csv', index=False)

    mask = t.group.notna()
    X = tr.loc[mask, feats]
    y = tr.loc[mask, 'Class_Name']
    groups = t.loc[mask, 'group']

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.empty(len(X), dtype=object)
    pred[:] = None
    rows = []

    for fold, (ix, jx) in enumerate(cv.split(X, y, groups), 1):
        m = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ).fit(X.iloc[ix], y.iloc[ix])
        pr = m.predict(X.iloc[jx])
        pred[jx] = pr

        rows.append({
            'fold': fold,
            'n_train': len(ix),
            'n_test': len(jx),
            'groups_train': groups.iloc[ix].nunique(),
            'groups_test': groups.iloc[jx].nunique(),
            'classes_test': ','.join(sorted(y.iloc[jx].unique())),
            'OA': accuracy_score(y.iloc[jx], pr),
            'BA': balanced_accuracy_score(y.iloc[jx], pr),
            'macroF1': f1_score(y.iloc[jx], pr, average='macro', zero_division=0),
        })

    print('folds', pd.DataFrame(rows).to_string(index=False))
    print(
        'pooled',
        accuracy_score(y, pred),
        balanced_accuracy_score(y, pred),
        f1_score(y, pred, average='macro'),
    )
    print('matrix', confusion_matrix(
        y, pred, labels=['Burned', 'Unburned', 'Man-made', 'Mixed']
    ))

    pd.DataFrame(rows).to_csv(P / 'ERST619_spatial_group_cv_folds.csv', index=False)
    out = t.loc[mask].copy()
    out['Predicted'] = pred
    out['Correct'] = y.values == pred
    out.to_csv(P / 'ERST619_spatial_cv_predictions.csv', index=False)

    print('note groups per class', t.groupby('Class_Name').group.nunique().to_dict())
    print('notes', t.groupby(['Class_Name', 'Notes']).agg(
        pixels=('pointid', 'count'),
        groups=('group', 'nunique'),
    ).to_string())
