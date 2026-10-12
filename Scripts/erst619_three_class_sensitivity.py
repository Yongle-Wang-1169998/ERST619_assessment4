"""ERST619: three-class Random Forest sensitivity analysis.

Compare an original four-class model to a three-class model on the *same*
329 non-Mixed validation pixels; run three-class polygon-grouped CV; produce
three-class classification GeoTIFF; optionally clip to a fire perimeter and
compare aligned old/new fire-extent rasters.

Requirements: numpy, pandas, scikit-learn, rasterio; optional geopandas for
--fire-boundary. Run: python erst619_three_class_sensitivity.py --data-dir D:\\619

Training data and evaluation design are derived from ERST619 project exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = ["dNBR", "B8", "NDVI", "B4"]
CLASSES_4 = ["Burned", "Unburned", "Man-made", "Mixed"]
CLASSES_3 = CLASSES_4[:3]
CODES = {"Burned": 1, "Unburned": 2, "Man-made": 3, "Mixed": 4}
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "dNBR"]
SEED = 42


def rf():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def save_matrix(y_true, y_pred, labels, path):
    arr = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(arr, index=pd.Index(labels, name="Actual"),
                 columns=pd.Index(labels, name="Predicted")).to_csv(path)


def classify_raster(source, outfile, model):
    """Honor band descriptions; classify only finite pixels; 0=NoData."""
    with rasterio.open(source) as src:
        desc = list(src.descriptions)
        if not all(x in desc for x in FEATURES):
            if src.count == 8:
                desc = BAND_NAMES
            else:
                raise ValueError(f"Cannot identify predictors in {source}; band descriptions={desc}")
        # Stack four predictors in the exact order used for model training.
        stacked = np.stack([src.read(desc.index(name) + 1) for name in FEATURES])
        valid = np.isfinite(stacked).all(axis=0)
        for i, name in enumerate(FEATURES):
            nodata = src.nodatavals[desc.index(name)]
            if nodata is not None:
                valid &= stacked[i] != nodata
        # If exported stack uses zero-filled missing observations, users MUST
        # check input validity and supply correct NoData metadata before use.
        pixels = stacked[:, valid].T
        output = np.zeros(src.shape, dtype=np.uint8)
        if len(pixels):
            X = pd.DataFrame(pixels, columns=FEATURES)
            output[valid] = np.array([CODES[x] for x in model.predict(X)], dtype=np.uint8)
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0,
                       compress="lzw", tiled=False)
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(output, 1)
            dst.set_band_description(1, "Three-class RF: 1 Burned, 2 Unburned, 3 Man-made")
    print(f"Saved full-extent 3-class GeoTIFF: {outfile} (valid pixels={valid.sum()})")


def clip_to_boundary(source, boundary_path, outfile):
    import geopandas as gpd
    boundary = gpd.read_file(boundary_path)
    with rasterio.open(source) as src:
        if boundary.crs is None:
            raise ValueError("Fire boundary lacks CRS; define it correctly in GIS first")
        boundary = boundary.to_crs(src.crs)
        shapes = [g.__geo_interface__ for g in boundary.geometry if g is not None and not g.is_empty]
        if not shapes:
            raise ValueError("No valid fire-boundary geometries")
        data, transform = raster_mask(src, shapes, crop=True, filled=True, nodata=0)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform,
                       nodata=0)
        with rasterio.open(outfile, "w", **profile) as dst:
            dst.write(data)
    print(f"Saved clipped GeoTIFF: {outfile}")


def area_and_transition(old_path, new_path, out_dir):
    with rasterio.open(old_path) as old, rasterio.open(new_path) as new:
        if old.crs != new.crs or old.transform != new.transform or old.shape != new.shape:
            raise ValueError("Old/new classified rasters are NOT on identical grids; align in ArcGIS Pro first")
        a, b = old.read(1), new.read(1)
        common = np.isin(a, [1, 2, 3, 4]) & np.isin(b, [1, 2, 3])
        pixel_ha = abs(old.transform.a * old.transform.e - old.transform.b * old.transform.d) / 10000
        if old.crs and old.crs.is_geographic:
            raise ValueError("Projected, metre-based raster CRS required for hectare area calculation")
        # One row per old/new class, including zero entries for ease of auditing.
        transitions = []
        for i in [1, 2, 3, 4]:
            for j in [1, 2, 3]:
                n = int(np.count_nonzero(common & (a == i) & (b == j)))
                transitions.append({"old_class": CLASSES_4[i-1], "new_class": CLASSES_3[j-1],
                                    "pixels": n, "area_ha": round(n*pixel_ha, 4)})
        pd.DataFrame(transitions).to_csv(out_dir / "ERST619_3class_transitions.csv", index=False)
        areas = []
        for mode, data, names in [("four_class", a, CLASSES_4), ("three_class", b, CLASSES_3)]:
            for code, name in enumerate(names, 1):
                n = int(np.count_nonzero(common & (data == code)))
                areas.append({"model":mode,"class":name,"pixels":n,
                              "area_ha":round(n*pixel_ha,4)})
        pd.DataFrame(areas).to_csv(out_dir / "ERST619_3class_area_comparison.csv", index=False)
        change = np.zeros(a.shape, dtype=np.uint8)
        change[common & (a == 4) & (b == 1)] = 1
        change[common & (a == 4) & (b == 2)] = 2
        change[common & (a == 4) & (b == 3)] = 3
        profile = old.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        out = out_dir / "ERST619_Mixed_reassignment.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(change, 1)
            dst.set_band_description(1, "1 Mixed to Burned; 2 Mixed to Unburned; 3 Mixed to Man-made")
        print(f"Common valid fire-extent pixels: {common.sum()}, area: {common.sum()*pixel_ha:.2f} ha")
        print("Mixed reassignment counts:",
              {CLASSES_3[i-1]:int(np.sum(change==i)) for i in [1,2,3]})
        print("Saved transition map:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--fire-boundary", type=Path,
                   help="Optional fire-perimeter shapefile/GeoPackage to clip three-class raster")
    p.add_argument("--old-clipped", type=Path,
                   help="Existing original four-class classification clipped to fire perimeter")
    p.add_argument("--new-clipped", type=Path,
                   help="Existing three-class classification clipped to SAME fire perimeter/grid")
    args = p.parse_args()
    root = args.data_dir.resolve()
    out = (args.output_dir or root / "ERST619_3class_results").resolve()
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "Training_Pixels_20m.csv")
    val = pd.read_csv(root / "Validation_Pixels_20m.csv")
    required = FEATURES + ["Class_Name", "pointid"]
    for name, df in [("Training", train), ("Validation", val)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")
    if train[FEATURES].isna().any().any() or val[FEATURES].isna().any().any():
        raise ValueError("CSV contains missing predictor values")
    tr3 = train[train.Class_Name.isin(CLASSES_3)].copy()
    va3 = val[val.Class_Name.isin(CLASSES_3)].copy()
    print(f"Training: 4-class {len(train)}, 3-class {len(tr3)}")
    print(f"Validation: 4-class {len(val)}, common non-Mixed subset {len(va3)}")

    original = rf().fit(train[FEATURES], train.Class_Name)
    simplified = rf().fit(tr3[FEATURES], tr3.Class_Name)
    metrics = []
    for name, model in [("four_class_same_subset", original), ("three_class_same_subset", simplified)]:
        pred = model.predict(va3[FEATURES])
        metrics.append({"evaluation":name, "n":len(va3),
                        "OA":accuracy_score(va3.Class_Name,pred),
                        "balanced_accuracy":balanced_accuracy_score(va3.Class_Name,pred),
                        "macro_F1":f1_score(va3.Class_Name,pred,average="macro",zero_division=0)})
        save_matrix(va3.Class_Name, pred, CLASSES_4 if name.startswith("four") else CLASSES_3,
                    out / f"ERST619_{name}_confusion.csv")
        print(name, classification_report(va3.Class_Name, pred, zero_division=0))
    fullpred = original.predict(val[FEATURES])
    metrics.append({"evaluation":"four_class_full_validation", "n":len(val),
                    "OA":accuracy_score(val.Class_Name,fullpred),
                    "balanced_accuracy":balanced_accuracy_score(val.Class_Name,fullpred),
                    "macro_F1":f1_score(val.Class_Name,fullpred,average="macro",zero_division=0)})
    pd.DataFrame(metrics).to_csv(out / "ERST619_3class_validation_comparison.csv",index=False)
    print("Validation metrics:\n",pd.DataFrame(metrics).to_string(index=False))

    # Map groups by pointid (never by row position) to protect against reordering.
    mapping = pd.read_csv(root / "ERST619_training_pixel_polygon_mapping.csv")
    if mapping.pointid.duplicated().any() or train.pointid.duplicated().any():
        raise ValueError("Duplicate pointid: cannot safely match training pixels to polygons")
    cvdata = tr3.merge(mapping[["pointid", "Class_Name", "group"]],
                       on=["pointid","Class_Name"], how="left", validate="one_to_one")
    matched = cvdata.dropna(subset=["group"]).reset_index(drop=True)
    print(f"Spatial CV: matched {len(matched)} / {len(tr3)} training pixels, "
          f"groups={matched.group.nunique()}")
    X, y, groups = matched[FEATURES], matched.Class_Name, matched.group
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    cvpred = np.empty(len(matched),dtype=object)
    folds=[]
    for fold,(ix,jx) in enumerate(cv.split(X,y,groups),1):
        m=rf().fit(X.iloc[ix], y.iloc[ix]); pr=m.predict(X.iloc[jx]); cvpred[jx]=pr
        overlap=set(groups.iloc[ix]) & set(groups.iloc[jx])
        if overlap:
            raise AssertionError(f"Group leakage in fold {fold}: {overlap}")
        folds.append({"fold":fold,"n_train":len(ix),"n_test":len(jx),
                      "n_train_groups":groups.iloc[ix].nunique(),
                      "n_test_groups":groups.iloc[jx].nunique(),
                      "test_class_counts":json.dumps(y.iloc[jx].value_counts().to_dict()),
                      "OA":accuracy_score(y.iloc[jx],pr),
                      "balanced_accuracy":balanced_accuracy_score(y.iloc[jx],pr),
                      "macro_F1":f1_score(y.iloc[jx],pr,average="macro",zero_division=0)})
    pd.DataFrame(folds).to_csv(out/"ERST619_3class_grouped_cv_folds.csv",index=False)
    pd.DataFrame({"pointid":matched.pointid,"group":groups,
                  "actual":y,"predicted":cvpred}).to_csv(
                      out/"ERST619_3class_grouped_cv_predictions.csv",index=False)
    save_matrix(y,cvpred,CLASSES_3,out/"ERST619_3class_grouped_cv_confusion.csv")
    print("Grouped CV pooled OA =",round(accuracy_score(y,cvpred)*100,2),"%")
    print(pd.DataFrame(folds).to_string(index=False))

    source = root / "Port_Hills_Classification_8Bands_2024.tif"
    full_tif = out / "Port_Hills_RF_3Classes.tif"
    classify_raster(source, full_tif, simplified)
    generated_clipped = None
    if args.fire_boundary:
        generated_clipped = out / "Port_Hills_RF_3Classes_Clipped.tif"
        clip_to_boundary(full_tif,args.fire_boundary,generated_clipped)
    new_clipped = args.new_clipped or generated_clipped
    if args.old_clipped and new_clipped:
        area_and_transition(args.old_clipped,new_clipped,out)
    else:
        print("No aligned old/new clipped raster pair supplied; "
              "area transition step skipped. Provide --old-clipped and --new-clipped.")
    print("Results folder:",out)
    print("CAUTION: non-Mixed validation excludes 150 Mixed-labelled pixels; "
          "higher OA does not by itself imply better landscape mapping.")


if __name__ == "__main__":
    main()
