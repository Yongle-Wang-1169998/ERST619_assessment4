"""
ERST619 Assessment 4 — Three-Class Random Forest Sensitivity Analysis
====================================================================

Purpose
-------
Evaluate the effect of removing the heterogeneous Mixed class from the
original four-class Random Forest classification of the 2024 Port Hills fire.
Compare models on the same non-Mixed validation pixels, assess polygon-grouped
cross-validation, and examine spatial changes in predicted land-cover classes.

Inputs
------
- Training_Pixels_20m.csv
- Validation_Pixels_20m.csv
- ERST619_training_pixel_polygon_mapping.csv (or --mapping-file)
- Port_Hills_Classification_8Bands_2024.tif
- Optional: fire boundary and aligned old/new clipped classification rasters

Outputs
-------
- Validation metrics and confusion matrices (CSV)
- Figure_B2_Three_Class_Confusion_Matrix.png (300 dpi)
- Polygon-grouped cross-validation metrics and predictions (CSV)
- Three-class Random Forest GeoTIFF
- Optional: classification area, transition CSVs, and Mixed reassignment GeoTIFF

Requirements
------------
numpy, pandas, scikit-learn, rasterio, matplotlib
Optional: geopandas (only when using --fire-boundary)

Example
-------
python erst619_three_class_sensitivity_updated.py --data-dir "D:\\619\\assessment3_training" \
    --mapping-file "D:\\619\\assessment4\\ERST619_training_pixel_polygon_mapping.csv"

Notes
-----
The same-subset comparison uses 329 non-Mixed validation pixels. It does not
measure performance on the 150 excluded Mixed-labelled reference pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures even when running without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
FEATURES = ["dNBR", "B8", "NDVI", "B4"]
CLASSES_4 = ["Burned", "Unburned", "Man-made", "Mixed"]
CLASSES_3 = ["Burned", "Unburned", "Man-made"]
CODES = {"Burned": 1, "Unburned": 2, "Man-made": 3, "Mixed": 4}
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "dNBR"]
SEED = 42


def build_random_forest():
    """Create the Random Forest used in the original classification."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    )


def save_confusion_matrix(y_true, y_pred, labels, output_csv):
    """Save raw-count confusion matrix with explicit class ordering."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    table = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )
    table.to_csv(output_csv)
    return matrix


def plot_three_class_confusion_matrix(y_true, y_pred, output_png):
    """Export the three-class validation matrix for Appendix B, Figure B2."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES_3)
    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES_3,
    )
    display.plot(
        ax=axis,
        cmap="viridis",
        values_format="d",
        colorbar=True,
    )
    axis.set_title("Three-Class Random Forest Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved three-class confusion matrix figure: {output_png}")
    return matrix


# -----------------------------------------------------------------------------
# 2. Raster classification and optional fire-boundary processing
# -----------------------------------------------------------------------------
def classify_raster(source, output_tif, model):
    """Classify pixels with finite predictors; output code 0 means NoData."""
    with rasterio.open(source) as src:
        band_descriptions = list(src.descriptions)
        if not all(name in band_descriptions for name in FEATURES):
            if src.count == 8:
                band_descriptions = BAND_NAMES
            else:
                raise ValueError(
                    f"Cannot identify predictor bands in {source}: "
                    f"{band_descriptions}"
                )

        stacked = np.stack(
            [src.read(band_descriptions.index(name) + 1) for name in FEATURES]
        )
        valid = np.isfinite(stacked).all(axis=0)
        for index, name in enumerate(FEATURES):
            nodata = src.nodatavals[band_descriptions.index(name)]
            if nodata is not None:
                valid &= stacked[index] != nodata

        # Review zero-filled background pixels before interpreting this raster.
        pixels = stacked[:, valid].T
        classification = np.zeros(src.shape, dtype=np.uint8)
        if len(pixels):
            predictors = pd.DataFrame(pixels, columns=FEATURES)
            classification[valid] = np.array(
                [CODES[label] for label in model.predict(predictors)],
                dtype=np.uint8,
            )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff", count=1, dtype="uint8", nodata=0,
            compress="lzw", tiled=False,
        )
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(classification, 1)
            dst.set_band_description(
                1, "Three-class RF: 1 Burned; 2 Unburned; 3 Man-made"
            )

    print(f"Saved three-class GeoTIFF: {output_tif} ({valid.sum()} valid pixels)")


def clip_to_boundary(source, boundary_file, output_tif):
    """Optionally clip the predicted raster to the approximate fire perimeter."""
    import geopandas as gpd

    boundary = gpd.read_file(boundary_file)
    with rasterio.open(source) as src:
        if boundary.crs is None:
            raise ValueError("Fire boundary has no CRS; define it in GIS first.")
        boundary = boundary.to_crs(src.crs)
        shapes = [
            geom.__geo_interface__
            for geom in boundary.geometry
            if geom is not None and not geom.is_empty
        ]
        if not shapes:
            raise ValueError("Fire boundary contains no valid geometries.")
        data, transform = raster_mask(
            src, shapes, crop=True, filled=True, nodata=0
        )
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1], width=data.shape[2],
            transform=transform, nodata=0,
        )
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(data)
    print(f"Saved clipped GeoTIFF: {output_tif}")


def compare_classified_areas(old_path, new_path, output_dir):
    """Compare pixel transitions only where old/new raster grids align."""
    with rasterio.open(old_path) as old, rasterio.open(new_path) as new:
        if (
            old.crs != new.crs
            or old.transform != new.transform
            or old.shape != new.shape
        ):
            raise ValueError(
                "Old and new clipped rasters do not share the same grid. "
                "Align them in ArcGIS Pro before comparison."
            )
        if old.crs is None or old.crs.is_geographic:
            raise ValueError("Use a projected raster CRS in metres for area estimates.")

        old_data = old.read(1)
        new_data = new.read(1)
        common = np.isin(old_data, [1, 2, 3, 4]) & np.isin(
            new_data, [1, 2, 3]
        )
        area_per_pixel_ha = abs(
            old.transform.a * old.transform.e
            - old.transform.b * old.transform.d
        ) / 10000

        transitions = []
        for old_code, old_name in enumerate(CLASSES_4, start=1):
            for new_code, new_name in enumerate(CLASSES_3, start=1):
                count = int(np.count_nonzero(
                    common & (old_data == old_code) & (new_data == new_code)
                ))
                transitions.append({
                    "old_class": old_name,
                    "new_class": new_name,
                    "pixels": count,
                    "area_ha": round(count * area_per_pixel_ha, 4),
                })
        pd.DataFrame(transitions).to_csv(
            output_dir / "ERST619_3class_transitions.csv", index=False
        )

        areas = []
        for model_name, data, labels in [
            ("four_class", old_data, CLASSES_4),
            ("three_class", new_data, CLASSES_3),
        ]:
            for code, label in enumerate(labels, start=1):
                count = int(np.count_nonzero(common & (data == code)))
                areas.append({
                    "model": model_name,
                    "class": label,
                    "pixels": count,
                    "area_ha": round(count * area_per_pixel_ha, 4),
                })
        pd.DataFrame(areas).to_csv(
            output_dir / "ERST619_3class_area_comparison.csv", index=False
        )

        reassignment = np.zeros(old_data.shape, dtype=np.uint8)
        for code in (1, 2, 3):
            reassignment[
                common & (old_data == 4) & (new_data == code)
            ] = code
        profile = old.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        output_tif = output_dir / "ERST619_Mixed_reassignment.tif"
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(reassignment, 1)
            dst.set_band_description(
                1,
                "1 Mixed to Burned; 2 Mixed to Unburned; 3 Mixed to Man-made",
            )

        print(
            f"Common fire-extent pixels: {common.sum()}, "
            f"area: {common.sum() * area_per_pixel_ha:.2f} ha"
        )
        print("Saved Mixed reassignment raster:", output_tif)


# -----------------------------------------------------------------------------
# 3. Model training, common-subset validation, and confusion matrix figure
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mapping-file", type=Path,
        help="Optional path to ERST619_training_pixel_polygon_mapping.csv",
    )
    parser.add_argument("--fire-boundary", type=Path)
    parser.add_argument("--old-clipped", type=Path)
    parser.add_argument("--new-clipped", type=Path)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    output_dir = (args.output_dir or root / "ERST619_3class_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(root / "Training_Pixels_20m.csv")
    validation = pd.read_csv(root / "Validation_Pixels_20m.csv")
    required = FEATURES + ["Class_Name", "pointid"]
    for label, frame in [("Training", training), ("Validation", validation)]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} CSV missing columns: {missing}")
        if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"{label} CSV contains invalid predictor values")

    training_3 = training[training.Class_Name.isin(CLASSES_3)].copy()
    validation_3 = validation[validation.Class_Name.isin(CLASSES_3)].copy()
    print(f"Training: four-class={len(training)}, three-class={len(training_3)}")
    print(f"Common non-Mixed validation subset: {len(validation_3)} pixels")

    original_rf = build_random_forest().fit(
        training[FEATURES], training.Class_Name
    )
    three_class_rf = build_random_forest().fit(
        training_3[FEATURES], training_3.Class_Name
    )

    comparisons = []
    for name, model, class_names in [
        ("four_class_same_subset", original_rf, CLASSES_4),
        ("three_class_same_subset", three_class_rf, CLASSES_3),
    ]:
        predictions = model.predict(validation_3[FEATURES])
        comparisons.append({
            "evaluation": name,
            "n": len(validation_3),
            "OA": accuracy_score(validation_3.Class_Name, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                validation_3.Class_Name, predictions
            ),
            "macro_F1": f1_score(
                validation_3.Class_Name, predictions,
                average="macro", zero_division=0,
            ),
        })
        save_confusion_matrix(
            validation_3.Class_Name,
            predictions,
            class_names,
            output_dir / f"ERST619_{name}_confusion.csv",
        )
        print(name)
        print(classification_report(
            validation_3.Class_Name, predictions, zero_division=0
        ))
        if name == "three_class_same_subset":
            plot_three_class_confusion_matrix(
                validation_3.Class_Name,
                predictions,
                output_dir / "Figure_B2_Three_Class_Confusion_Matrix.png",
            )

    full_predictions = original_rf.predict(validation[FEATURES])
    comparisons.append({
        "evaluation": "four_class_full_validation",
        "n": len(validation),
        "OA": accuracy_score(validation.Class_Name, full_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            validation.Class_Name, full_predictions
        ),
        "macro_F1": f1_score(
            validation.Class_Name, full_predictions,
            average="macro", zero_division=0,
        ),
    })
    pd.DataFrame(comparisons).to_csv(
        output_dir / "ERST619_3class_validation_comparison.csv", index=False
    )
    print("Validation comparison:")
    print(pd.DataFrame(comparisons).to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Polygon-grouped cross-validation
    # -------------------------------------------------------------------------
    mapping_path = Path(r"D:\619\assessment4\ERST619_training_pixel_polygon_mapping.csv")

    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Polygon mapping not found: {mapping_path}. "
            "Use --mapping-file to specify its location."
        )
    mapping = pd.read_csv(mapping_path)
    if mapping.pointid.duplicated().any() or training.pointid.duplicated().any():
        raise ValueError("Duplicate pointid; cannot match pixels to polygons")
    cv_data = training_3.merge(
        mapping[["pointid", "Class_Name", "group"]],
        on=["pointid", "Class_Name"],
        how="left",
        validate="one_to_one",
    )
    matched = cv_data.dropna(subset=["group"]).reset_index(drop=True)
    print(
        f"Polygon-grouped CV: {len(matched)}/{len(training_3)} pixels, "
        f"{matched.group.nunique()} groups"
    )
    predictors = matched[FEATURES]
    labels = matched.Class_Name
    groups = matched.group
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    cv_predictions = np.empty(len(matched), dtype=object)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(predictors, labels, groups), start=1
    ):
        model = build_random_forest().fit(
            predictors.iloc[train_idx], labels.iloc[train_idx]
        )
        predictions = model.predict(predictors.iloc[test_idx])
        cv_predictions[test_idx] = predictions
        if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
            raise AssertionError(f"Group leakage in fold {fold}")
        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_groups": groups.iloc[train_idx].nunique(),
            "n_test_groups": groups.iloc[test_idx].nunique(),
            "test_class_counts": json.dumps(
                labels.iloc[test_idx].value_counts().to_dict()
            ),
            "OA": accuracy_score(labels.iloc[test_idx], predictions),
            "balanced_accuracy": balanced_accuracy_score(
                labels.iloc[test_idx], predictions
            ),
            "macro_F1": f1_score(
                labels.iloc[test_idx], predictions,
                average="macro", zero_division=0,
            ),
        })

    pd.DataFrame(fold_results).to_csv(
        output_dir / "ERST619_3class_grouped_cv_folds.csv", index=False
    )
    pd.DataFrame({
        "pointid": matched.pointid,
        "group": groups,
        "actual": labels,
        "predicted": cv_predictions,
    }).to_csv(
        output_dir / "ERST619_3class_grouped_cv_predictions.csv", index=False
    )
    save_confusion_matrix(
        labels,
        cv_predictions,
        CLASSES_3,
        output_dir / "ERST619_3class_grouped_cv_confusion.csv",
    )
    print(f"Grouped CV pooled OA: {accuracy_score(labels, cv_predictions):.2%}")
    print(pd.DataFrame(fold_results).to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Export classification and optionally calculate transition areas
    # -------------------------------------------------------------------------
    source_raster = root / "Port_Hills_Classification_8Bands_2024.tif"
    full_raster = output_dir / "Port_Hills_RF_3Classes.tif"
    classify_raster(source_raster, full_raster, three_class_rf)

    clipped_raster = None
    if args.fire_boundary:
        clipped_raster = output_dir / "Port_Hills_RF_3Classes_Clipped.tif"
        clip_to_boundary(full_raster, args.fire_boundary, clipped_raster)
    new_clipped = args.new_clipped or clipped_raster
    if args.old_clipped and new_clipped:
        compare_classified_areas(args.old_clipped, new_clipped, output_dir)
    else:
        print(
            "Area transition comparison skipped: provide --old-clipped "
            "and --new-clipped (or --fire-boundary)."
        )

    print("Results folder:", output_dir)
    print(
        "CAUTION: excluding the 150 Mixed validation pixels changes the "
        "classification task; higher OA alone does not prove better mapping."
    )


if __name__ == "__main__":
    main()
