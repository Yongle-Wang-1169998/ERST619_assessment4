
import arcpy
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

raster_path = r"D:\619\assessment3\Port_Hills_Classification_8Bands_2024.tif"

# 修改为你实际的 TIFF 文件路径
raster = arcpy.Raster(raster_path)

# 转换为 NumPy 数组
data = arcpy.RasterToNumPyArray(raster).astype("float64")

print("Raster array shape:", data.shape)

# 多波段影像通常为 (bands, rows, columns)
band_names = [
    "B2", "B3", "B4", "B8",
    "B11", "B12", "NDVI", "dNBR"
]

# 仅保留八个波段全部有效的像元
valid_mask = np.all(np.isfinite(data), axis=0)

# 如源影像有明确的 NoData 值，还需要排除这些值
for i in range(data.shape[0]):
    nodata = raster.getRasterBands()[i].noDataValue
    if nodata is not None:
        valid_mask &= (data[i] != nodata)

records = []

for i, name in enumerate(band_names):
    values = data[i][valid_mask]

    records.append({
        "Variable": name,
        "Mean": np.mean(values),
        "Median": np.median(values),
        "SD": np.std(values, ddof=1),
        "Skewness": skew(values, bias=False),
        "Kurtosis": kurtosis(values, fisher=True, bias=False)
    })

table = pd.DataFrame(records)

print("Valid pixels:", valid_mask.sum())
print(table.round(4).to_string(index=False))


table.to_csv(
    r"D:\619\assessment4\Table_A1_Distribution_Statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Table A1 CSV saved successfully.")

