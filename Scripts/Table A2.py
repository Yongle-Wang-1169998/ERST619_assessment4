
import pandas as pd

# 
df = pd.read_csv(r"D:\619\assessment4\Training_Pixels_20m.csv")

# 
print(df.columns.tolist())

features = [
    "B2", "B3", "B4", "B8",
    "B11", "B12", "NDVI", "dNBR"
]

# 根据实际 CSV 修改类别字段名称
class_col = "Class_Name"

results = []

for class_name, group in df.groupby(class_col):
    flagged = pd.DataFrame(index=group.index)

    for feature in features:
        q1 = group[feature].quantile(0.25)
        q3 = group[feature].quantile(0.75)
        iqr = q3 - q1

        outlier = (
            (group[feature] < q1 - 1.5 * iqr) |
            (group[feature] > q3 + 1.5 * iqr)
        )

        flagged[feature] = outlier

    unique_outliers = flagged.any(axis=1).sum()

    results.append({
        "Class": class_name,
        "Total pixels": len(group),
        **{f: int(flagged[f].sum()) for f in features},
        "Unique outlier pixels": int(unique_outliers),
        "Percentage": round(
            unique_outliers / len(group) * 100, 2
        )
    })

result = pd.DataFrame(results)
print(result.to_string(index=False))
