import os
import sys
import subprocess
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
from rasterio.warp import reproject, Resampling
from affine import Affine
from osgeo import gdal



def setup_environment():
    
    required_packages = ['rasterio', 'pyproj', 'affine']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Packages installed successfully. Please restart the script.")
        sys.exit(0)


setup_environment()


try:
    import pyproj
    from osgeo import gdal, osr
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install GDAL using conda: conda install -c conda-forge gdal")
    sys.exit(1)

os.environ['SIXS_DIR'] = r'"C:\Users\ujjwal\OneDrive\Desktop\Projects\Anant_Task2\6S_code\6S\build"'

def load_dark_frame(dark_frame_path, target_shape=None):
    
    with rasterio.open(dark_frame_path) as src:
        dark_frame = src.read().astype(np.float32)
    
    if target_shape and dark_frame.shape != target_shape:
        
        resized_dark = np.empty(target_shape, dtype=np.float32)
        for band in range(dark_frame.shape[0]):
            resized_dark[band] = cv2.resize(dark_frame[band], 
                                           (target_shape[2], target_shape[1]))
        return resized_dark
    return dark_frame

def create_master_dark(dark_frame_paths):
    
    if not dark_frame_paths:
        return None
        
    dark_frames = []
    for path in dark_frame_paths:
        with rasterio.open(path) as src:
            dark_frames.append(src.read().astype(np.float32))
    
    
    master_dark = np.mean(np.array(dark_frames), axis=0)
    return master_dark

def create_synthetic_dark_frame(image_shape, hot_pixel_density=0.01, noise_level=5.0):
   
    
    bands, height, width = image_shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    
    dist_from_center = np.sqrt(((y - center_y) / center_y) ** 2 + 
                              ((x - center_x) / center_x) ** 2)
    
    
    dark_frame = np.empty(image_shape, dtype=np.float32)
    for band in range(bands):
        
        band_noise = noise_level * (0.8 + 0.4 * (band / bands))
        
        noise_pattern = band_noise * (0.5 + 0.5 * dist_from_center)
        dark_frame[band] = noise_pattern
    
    
    num_hot_pixels = int(hot_pixel_density * height * width)
    for band in range(bands):
        hot_y = np.random.randint(0, height, num_hot_pixels)
        hot_x = np.random.randint(0, width, num_hot_pixels)
        hot_values = np.random.uniform(noise_level*2, noise_level*5, num_hot_pixels)
        for i in range(num_hot_pixels):
            dark_frame[band, hot_y[i], hot_x[i]] = hot_values[i]
    
    return dark_frame

def dark_frame_subtraction(image, dark_frame):
    
    
    if dark_frame.shape != image.shape:
        raise ValueError("Dark frame dimensions must match image dimensions")
    
    
    corrected = image - dark_frame
    
    
    return np.clip(corrected, 0, None)

def estimate_flat_field(image, method="gaussian", kernel_size=101):

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
    
    flat_mean = np.mean(flat_field, axis=(1, 2), keepdims=True)
    return image * (flat_mean / (flat_field + 1e-6))

class DnCNN(nn.Module):
    
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
    
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).repeat(1, 13, 1, 1).float()
        denoised_tensor = model(image_tensor)
        return denoised_tensor.squeeze()[0].numpy()

def hybrid_denoise(image, model):
    
    denoised = np.empty_like(image)
    for band in range(image.shape[0]):
        denoised[band] = dncnn_denoise(image[band], model)
    return denoised.astype(np.float32)

def toa_to_sr_6s(image, solar_angle):
    
    s = SixS("C:/Users/ujjwal/OneDrive/Desktop/Projects/Anant_Task2/6S_code/6S/build/sixs.exe")
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Maritime)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.1)
    s.geometry = Geometry.User()
    s.geometry.solar_z = solar_angle
    s.run()
    
    return image / s.outputs.transmittance_total_scattering.total

def compute_indices(image):
    
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

def load_rpc_model(rpc_file):
    
    try:
        
        ds = gdal.Open(rpc_file)
        if ds is None:
            print(f"Could not open {rpc_file}")
            return None
        
        
        rpc_data = {}
        metadata = ds.GetMetadata('RPC')
        
        if not metadata:
            print(f"No RPC metadata found in {rpc_file}")
            return None
            
        
        for key, value in metadata.items():
            if key in ['LINE_NUM_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']:
                rpc_data[key] = [float(x) for x in value.split()]
            else:
                try:
                    rpc_data[key] = float(value)
                except ValueError:
                    rpc_data[key] = value
                    
        ds = None  
        return rpc_data
    except Exception as e:
        print(f"Error loading RPC file: {e}")
        return None

def load_dem(dem_path, bounds=None):
   
    try:
        with rasterio.open(dem_path) as src:
            if bounds:
                
                window = src.window(*bounds)
                dem = src.read(1, window=window)
            else:
                dem = src.read(1)
            transform = src.transform
            crs = src.crs
        return dem, transform, crs
    except Exception as e:
        print(f"Error loading DEM: {e}")
        return None, None, None

def apply_rpc_model(lat, lon, height, rpc_data):
    
    lat_norm = (lat - rpc_data['LAT_OFF']) / rpc_data['LAT_SCALE']
    lon_norm = (lon - rpc_data['LONG_OFF']) / rpc_data['LONG_SCALE']
    height_norm = (height - rpc_data['HEIGHT_OFF']) / rpc_data['HEIGHT_SCALE']
    
    
    terms = [1.0, lon_norm, lat_norm, height_norm, lon_norm*lat_norm,
             lon_norm*height_norm, lat_norm*height_norm, lon_norm**2, lat_norm**2,
             height_norm**2, lon_norm*lat_norm*height_norm, lon_norm**3,
             lon_norm*lat_norm**2, lon_norm*height_norm**2, lon_norm**2*lat_norm,
             lat_norm**3, lat_norm*height_norm**2, lon_norm**2*height_norm,
             lat_norm**2*height_norm, height_norm**3]
    
    
    line_num = sum(rpc_data['LINE_NUM_COEFF'][i] * terms[i] for i in range(20))
    line_den = sum(rpc_data['LINE_DEN_COEFF'][i] * terms[i] for i in range(20))
    samp_num = sum(rpc_data['SAMP_NUM_COEFF'][i] * terms[i] for i in range(20))
    samp_den = sum(rpc_data['SAMP_DEN_COEFF'][i] * terms[i] for i in range(20))
    
    
    line = line_num / line_den
    samp = samp_num / samp_den
    
    
    line = line * rpc_data['LINE_SCALE'] + rpc_data['LINE_OFF']
    samp = samp * rpc_data['SAMP_SCALE'] + rpc_data['SAMP_OFF']
    
    return line, samp

def orthorectify_image(image, src_meta, dem_path, rpc_path, output_resolution=10.0):
    
    
    dem, dem_transform, dem_crs = load_dem(dem_path)
    if dem is None:
        print("Failed to load DEM. Skipping orthorectification.")
        return image, src_meta
    
    
    rpc_data = load_rpc_model(rpc_path)
    if rpc_data is None:
        print("Failed to load RPC model. Skipping orthorectification.")
        return image, src_meta
    
    
    height, width = image.shape[1], image.shape[2]
    bands = image.shape[0]
    
    
    dst_crs = dem_crs
    
    
    dst_transform = Affine(output_resolution, 0.0, rpc_data.get('LONG_OFF', 0) - width/2*output_resolution,
                          0.0, -output_resolution, rpc_data.get('LAT_OFF', 0) + height/2*output_resolution)
    
    
    ortho_image = np.zeros((bands, height, width), dtype=np.float32)
    
    
    for y in range(height):
        for x in range(width):
            
            lon, lat = dst_transform * (x, y)
            
            
            row, col = ~dem_transform * (lon, lat)
            row, col = int(row), int(col)
            if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                elevation = dem[row, col]
            else:
                elevation = 0  
            
            
            line, samp = apply_rpc_model(lat, lon, elevation, rpc_data)
            
            
            line_int, samp_int = int(line), int(samp)
            if 0 <= line_int < height and 0 <= samp_int < width:
                
                for b in range(bands):
                    ortho_image[b, y, x] = image[b, line_int, samp_int]
    
    
    ortho_meta = src_meta.copy()
    ortho_meta.update({
        'crs': dst_crs,
        'transform': dst_transform,
        'width': width,
        'height': height
    })
    
    return ortho_image, ortho_meta

def full_correction_pipeline(input_path, dark_frame_path=None, dark_frame_paths=None, 
                            use_synthetic_dark=False, dncnn_model=None, 
                            flat_field_method="gaussian", solar_angle=30,
                            dem_path=None, rpc_path=None, apply_orthorectification=False):
    "
    
    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)
        src_meta = src.meta.copy()
    
    
    if dark_frame_path:
        
        dark_frame = load_dark_frame(dark_frame_path, image.shape)
        image = dark_frame_subtraction(image, dark_frame)
    elif dark_frame_paths:
    
        master_dark = create_master_dark(dark_frame_paths)
        image = dark_frame_subtraction(image, master_dark)

        



