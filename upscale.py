import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for denoising
from osgeo import gdal, gdalconst

##############################################################################
# CONFIGURATION: Input / Output Paths
##############################################################################
sat_file = "sat.tif"  # 4-band (B, G, R, NIR), 16-bit
lidar_file = "DSM_TQ0075_P_12757_20230109_20230315.tif"

sat_resampled = "sat_resampled.tif"      # Satellite to LiDAR resolution
lidar_hillshade = "lidar_hillshade.tif"  # Hillshade from LiDAR DSM
fused_output = "sat_lidar_pansharpen.tif"

##############################################################################
# STEP 1: Read LiDAR
##############################################################################
lidar_ds = gdal.Open(lidar_file, gdalconst.GA_ReadOnly)
if lidar_ds is None:
    raise RuntimeError(f"Failed to open LiDAR file: {lidar_file}")

lidar_gt = lidar_ds.GetGeoTransform()
lidar_xres = lidar_gt[1]
lidar_yres = lidar_gt[5]

sat_ds = gdal.Open(sat_file, gdalconst.GA_ReadOnly)
if sat_ds is None:
    raise RuntimeError(f"Failed to open satellite file: {sat_file}")

warp_options = gdal.WarpOptions(
    xRes=lidar_xres,
    yRes=abs(lidar_yres),
    resampleAlg="lanczos"
)
print("Warping satellite to match LiDAR resolution...")
gdal.Warp(sat_resampled, sat_ds, options=warp_options)

##############################################################################
# STEP 2: Create a LiDAR Hillshade
##############################################################################
print("Generating hillshade from LiDAR...")
gdal.DEMProcessing(
    lidar_hillshade,
    lidar_file,
    "hillshade",
    azimuth=315.0,
    altitude=45.0,
    scale=1.0,
    zFactor=1.0
)

##############################################################################
# STEP 3: Read Resampled Satellite & Hillshade
##############################################################################
print("Reading resampled satellite...")
sat_res_ds = gdal.Open(sat_resampled, gdalconst.GA_ReadOnly)
sat_res_data = sat_res_ds.ReadAsArray().astype(np.float32)

print("Reading LiDAR hillshade...")
hill_ds = gdal.Open(lidar_hillshade, gdalconst.GA_ReadOnly)
hill_data = hill_ds.ReadAsArray().astype(np.float32)

##############################################################################
# STEP 4: Percentile-Based Scaling of 16-bit Satellite B, G, R
##############################################################################
# Extract B, G, R, N
B = sat_res_data[0, :, :]
G = sat_res_data[1, :, :]
R = sat_res_data[2, :, :]
NIR = sat_res_data[3, :, :]

def percentile_range_scale(arr_list, lower_p=30, upper_p=70):
    """
    1) Flatten all arrays into one big array.
    2) Compute the lower and upper percentile across them.
    3) Scale each array to [0..1] using those percentile cutoffs, clipping outliers.
    """
    combined = np.concatenate([a.flatten() for a in arr_list])
    low_val = np.percentile(combined, lower_p)
    high_val = np.percentile(combined, upper_p)
    print(f"Global {lower_p}th percentile={low_val}, {upper_p}th percentile={high_val}")
    
    def scale(arr, lval, hval):
        arr_clipped = np.clip(arr, lval, hval)
        return (arr_clipped - lval) / (hval - lval + 1e-9)
    
    return [scale(a, low_val, high_val) for a in arr_list]

B_01, G_01, R_01 = percentile_range_scale([B, G, R], lower_p=2, upper_p=98)

#NIR scaling
N_min, N_max = NIR.min(), NIR.max()
NIR_01 = (NIR - N_min) / (N_max - N_min + 1e-9)

# satellite intensity from R, G, B
I_sat = 0.299 * R_01 + 0.587 * G_01 + 0.114 * B_01

##############################################################################
# STEP 5: Normalize Hillshade to [0..1], Then Combine with Satellite
##############################################################################
min_hill, max_hill = hill_data.min(), hill_data.max()
hill_01 = (hill_data - min_hill) / (max_hill - min_hill + 1e-9)

alpha = 0.5  # how much satellite intensity vs LiDAR
I_new = alpha * I_sat + (1.0 - alpha) * hill_01
ratio = I_new / (I_sat + 1e-9)  # Brovey-like ratio

# Sharpen
R_sharp = np.clip(R_01 * ratio, 0, 1)
G_sharp = np.clip(G_01 * ratio, 0, 1)
B_sharp = np.clip(B_01 * ratio, 0, 1)
NIR_sharp = np.clip(NIR_01 * ratio, 0, 1)

##############################################################################
# STEP 6:Gamma Correction
##############################################################################
gamma = 1
def gamma_correct(arr, gamma):
    return arr ** gamma

R_gamma = gamma_correct(R_sharp, gamma)
G_gamma = gamma_correct(G_sharp, gamma)
B_gamma = gamma_correct(B_sharp, gamma)
NIR_gamma = gamma_correct(NIR_sharp, gamma)

##############################################################################
# STEP 7: Convert to 8-bit, Denoise using fastNlMeansDenoisingColored, Combine Channels, Write Output
##############################################################################
# Convert gamma-corrected channels to 8-bit
R_8 = (R_gamma * 255).astype(np.uint8)
G_8 = (G_gamma * 255).astype(np.uint8)
B_8 = (B_gamma * 255).astype(np.uint8)
NIR_8 = (NIR_gamma * 255).astype(np.uint8)

# Merge B, G, R into a 3-channel image
bgr_img = cv2.merge([B_8, G_8, R_8])

# Apply non-local means denoising to reduce color artifacts
denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr_img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

# Split the denoised image back into B, G, R channels
B_8_denoised, G_8_denoised, R_8_denoised = cv2.split(denoised_bgr)

fused_data = np.stack([B_8_denoised, G_8_denoised, R_8_denoised, NIR_8], axis=0)

bands, rows, cols = fused_data.shape
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(
    fused_output,
    cols,
    rows,
    bands,
    gdalconst.GDT_Byte
)

# Copy projection & geotransform from the satellite
out_ds.SetProjection(sat_res_ds.GetProjection())
out_ds.SetGeoTransform(sat_res_ds.GetGeoTransform())

for i in range(bands):
    out_band = out_ds.GetRasterBand(i+1)
    out_band.WriteArray(fused_data[i])
    out_band.SetNoDataValue(0)
out_ds.FlushCache()
out_ds = None

print(f"Saved fused 4-band TIFF: {fused_output}")

sat_res_ds, hill_ds, lidar_ds, sat_ds = None, None, None, None

##############################################################################
# STEP 8: PLOT and SAVE
##############################################################################
rgb_vis = np.stack([
    fused_data[2, :, :],  # R
    fused_data[1, :, :],  # G
    fused_data[0, :, :]   # B
], axis=-1)

plt.imsave("sat_lidar_pansharpen.jpg", rgb_vis)

plt.figure(figsize=(10, 6))
plt.imshow(rgb_vis)
plt.title("Pan-sharpened (R, G, B) with Percentile Scaling, Gamma Correction, and NL Means Denoising")
plt.axis("off")
plt.show()
