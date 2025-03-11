import os
import cv2
import numpy as np
import rasterio
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from tqdm import tqdm
from Py6S import SixS, AtmosProfile, AeroProfile, GroundReflectance, Geometry


os.environ['SIXS_DIR'] = r'"C:\Users\ujjwal\OneDrive\Desktop\Projects\Anant_Task2\6S_code\6S\build"'




def estimate_flat_field(image, method="gaussian", kernel_size=101):
    """Estimates a flat field image using different methods."""
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "polynomial":
        h, w = image.shape[1:]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_data, y_data = x.ravel(), y.ravel()
        z_data = image.mean(axis=0).ravel()

        def polynomial_surface(xy, a, b, c, d, e, f):
            x, y = xy
            return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

        popt, _ = curve_fit(polynomial_surface, (x_data, y_data), z_data)
        flat_field = polynomial_surface((x, y), *popt).reshape(h, w)
        return np.tile(flat_field, (image.shape[0], 1, 1))
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'polynomial'.")

def flat_field_correction(image, flat_field):
    """Applies Flat Field Correction (FFC) to an image."""
    flat_mean = np.mean(flat_field, axis=(1, 2), keepdims=True)
    return image * (flat_mean / (flat_field + 1e-6))


class DnCNN(nn.Module):
    """Implements the DnCNN model for denoising radiometric errors."""
    def __init__(self, num_channels=1, num_layers=17):
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        return x - self.dncnn(x)

def dncnn_denoise(image, model):
    """Uses a pretrained DnCNN model to remove radiometric noise."""
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).repeat(1, 13, 1, 1).float()
        denoised_tensor = model(image_tensor)
    return denoised_tensor.squeeze()[0].numpy()


def hybrid_denoise(image, model):
    """Applies DnCNN-based denoising to each band separately."""
    denoised = np.empty_like(image)
    for band in range(image.shape[0]):
        denoised[band] = dncnn_denoise(image[band], model)
    return denoised.astype(np.float32)


def toa_to_sr_6s(image, solar_angle):
    """Converts TOA reflectance to Surface Reflectance using the 6S model."""
    
    s = SixS("C:/Users/ujjwal/OneDrive/Desktop/Projects/Anant_Task2/6S_code/6S/build/sixs.exe")
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Maritime)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.1)
    s.geometry = Geometry.User()
    s.geometry.solar_z = solar_angle
    s.run()
    return image / s.outputs.transmittance_total_scattering.total




def compute_indices(image):
    """Computes NDVI, NDWI, SAVI, MNDWI, and NDBI indices."""
    red = image[2]
    nir = image[3]
    swir1 = image[4]
    swir2 = image[5]
    green = image[1]
    
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
    mndwi = (green - swir1) / (green + swir1 + 1e-6)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    
    return {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "SAVI": savi,
        "MNDWI": mndwi,
        "NDBI": ndbi
    }


def generate_binary_masks(indices):
    """Generates thresholded binary masks highlighting vegetation, water bodies, soil, and urban areas."""
    vegetation_mask = indices["NDVI"] > 0.3
    water_mask = indices["NDWI"] > 0.2
    soil_mask = (indices["SAVI"] > 0.2) & (indices["NDVI"] < 0.3)
    urban_mask = indices["NDBI"] > 0.2
    
    return {
        "Vegetation": vegetation_mask,
        "Water": water_mask,
        "Soil": soil_mask,
        "Urban": urban_mask
    }


def full_correction_pipeline(input_path, dncnn_model=None, flat_field_method="gaussian", solar_angle=30):
    """Integrated pipeline: Applies FFC -> DnCNN-Based Radiometric Error Correction -> TOA to SR Conversion -> Computes Indices -> Generates Masks."""
    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)
    
    flat_field = estimate_flat_field(image, method=flat_field_method)
    ffc_corrected = flat_field_correction(image, flat_field)
    denoised_image = hybrid_denoise(ffc_corrected, dncnn_model) if dncnn_model else ffc_corrected
    surface_reflectance = toa_to_sr_6s(denoised_image, solar_angle)
    indices = compute_indices(surface_reflectance)
    masks = generate_binary_masks(indices)

    return indices, masks


dataset_path = r"C:\Users\ujjwal\OneDrive\Desktop\Projects\Anant_Task2\EuroSAT"
all_image_paths = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.tif') or file.endswith('.jpg'):
            all_image_paths.append(os.path.join(root, file))
print(f"Number of images found: {len(all_image_paths)}")
print(f"Dataset path exists: {os.path.exists(dataset_path)}")
if len(all_image_paths) > 0:
    print(f"Sample image path: {all_image_paths[0]}")
else:
    print(f"No images found in {os.path.abspath(dataset_path)}")

random.seed(42)
sampled_paths = random.sample(all_image_paths, min(len(all_image_paths), 500))

dncnn_model = DnCNN(num_channels=13)
dncnn_model.eval()

results = {}
for image_path in tqdm(sampled_paths, desc="Processing Images"):
    indices, masks = full_correction_pipeline(image_path, dncnn_model=dncnn_model)
    results[image_path] = {"indices": indices, "masks": masks}

print(f"Processed {len(results)} images.")

print(f"\nIndices for {os.path.basename(image_path)}:")
for index_name, index_value in indices.items():
    print(f"{index_name}: Min={np.min(index_value):.4f}, Max={np.max(index_value):.4f}, Mean={np.mean(index_value):.4f}")


output_dir = "output_masks"
os.makedirs(output_dir, exist_ok=True)
for mask_name, mask in masks.items():
    mask_image = mask.astype(np.uint8) * 255
    mask_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{mask_name}.png")
    cv2.imwrite(mask_path, mask_image)
    print(f"Saved {mask_name} mask to {mask_path}")


indices_dir = "output_indices"
os.makedirs(indices_dir, exist_ok=True)
for index_name, index_value in indices.items():
    plt.figure(figsize=(8, 8))
    plt.imshow(index_value, cmap='viridis')
    plt.colorbar(label=index_name)
    plt.title(f"{index_name} - {os.path.basename(image_path)}")
    index_path = os.path.join(indices_dir, f"{os.path.basename(image_path).split('.')[0]}_{index_name}.png")
    plt.savefig(index_path)
    plt.close()
    print(f"Saved {index_name} visualization to {index_path}")

