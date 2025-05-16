import numpy as np
import mrcfile
import os
import glob
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate, affine_transform, map_coordinates as ndimage_map_coordinates
from scipy import interpolate, ndimage
from scipy.spatial.transform import Rotation
import skimage.transform
from skimage import measure
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import noise
import subprocess
import tempfile
import Bio
from Bio.PDB import PDBParser, Selection, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import sys

def lin_map(x, lb=0, ub=1):
    min_x, max_x = np.min(x), np.max(x)
    if max_x == min_x:
        return np.ones_like(x) * 0.5 * (lb + ub)
    return lb + (x - min_x) * (ub - lb) / (max_x - min_x)

def load_mrc(path, mmap=True, no_saxes=True):
    with mrcfile.open(path, permissive=True, header_only=False) as mrc:
        arr = mrc.data
    if no_saxes:
        arr = np.squeeze(arr)
    return arr

def save_mrc(data, output_path, spacing=1.0):
    """
    Save data as MRC file
    
    Args:
        data: Data array to save
        output_path: Output file path
        spacing: Voxel spacing (Angstrom)
    """
    if len(data.shape) == 3:
        data_transposed = np.transpose(data, (2, 1, 0))
    else:
        data_transposed = data
        
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data_transposed.astype(np.float32))
        mrc.voxel_size = spacing
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3

def detect_scale(particle_size, volume_shape, pixel_size):
    """
    Automatically detect appropriate scale level
    
    Args:
        particle_size: Particle size (voxel units)
        volume_shape: Volume shape
        pixel_size: Pixel size
        
    Returns:
        scale_factor: Scale factor between 0.0-1.0
    """
    particle_physical_size = particle_size * pixel_size
    volume_physical_size = max(volume_shape) * pixel_size
    relative_size = particle_physical_size / volume_physical_size
    
    if particle_physical_size < 10:
        size_factor = 1.0
    elif particle_physical_size < 50:
        size_factor = 0.8
    elif particle_physical_size < 200:
        size_factor = 0.6
    elif particle_physical_size < 1000:
        size_factor = 0.4
    else:
        size_factor = 0.2
    
    volume_size = np.prod(volume_shape)
    density_factor = min(1.0, particle_size**3 / (volume_size / 1000))
    
    scale_factor = 0.7 * size_factor + 0.3 * density_factor
    scale_factor = min(1.0, max(0.0, scale_factor))
    
    scale_description = ""
    if scale_factor > 0.8:
        scale_description = "Nano scale (fine details, strict collision detection)"
    elif scale_factor > 0.5:
        scale_description = "Micro scale (good details, moderately strict collision detection)"
    elif scale_factor > 0.25:
        scale_description = "Meso scale (medium details, balanced collision detection)"
    else:
        scale_description = "Macro scale (simplified details, relaxed collision detection)"
    
    print(f"Auto-detected scale: {scale_factor:.2f} - {scale_description}")
    print(f"Particle size: {particle_physical_size:.1f}Å, Volume size: {volume_physical_size:.1f}Å")
    
    return scale_factor

def iso_surface(volume, threshold, closed=False, normals=None, scale_factor=0.5):
    """
    Extract isosurface from volume data
    
    Args:
        volume: Input volume data
        threshold: Isosurface threshold
        closed: Close the surface
        normals: Compute normals
        scale_factor: Scale factor
    
    Returns:
        vtkPolyData object
    """
    step_size = max(1, int(3 * (1.0 - scale_factor) + 1))
    smoothing_iterations = int(max(1, 5 * scale_factor))
    
    contour = measure.marching_cubes(volume, threshold, step_size=step_size)
    points, faces = contour[0], contour[1]
    
    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    
    for point in points:
        vtk_points.InsertNextPoint(*point)
    
    for face in faces:
        triangle = vtk.vtkTriangle()
        for i in range(3):
            triangle.GetPointIds().SetId(i, face[i])
        vtk_cells.InsertNextCell(triangle)
    
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)
    
    if smoothing_iterations > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(poly_data)
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetRelaxationFactor(0.2)
        smoother.Update()
        poly_data = smoother.GetOutput()
    
    if normals is not None:
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(poly_data)
        norms.ComputePointNormalsOn()
        norms.ComputeCellNormalsOff()
        norms.SplittingOff()
        norms.Update()
        poly_data = norms.GetOutput()
    
    if closed:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(poly_data)
        cleaner.Update()
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(cleaner.GetOutput())
        filler.Update()
        poly_data = filler.GetOutput()
    
    if scale_factor < 0.5:
        decimation_factor = 0.6 * (1.0 - scale_factor)
        if decimation_factor > 0.1:
            decimation = vtk.vtkDecimatePro()
            decimation.SetInputData(poly_data)
            decimation.SetTargetReduction(decimation_factor)
            decimation.PreserveTopologyOn()
            decimation.Update()
            poly_data = decimation.GetOutput()
    
    return poly_data

def poly_translate(poly, offset):
    transform = vtk.vtkTransform()
    transform.Translate(*offset)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    return transform_filter.GetOutput()

def poly_rotate(poly, quaternion):
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    return transform_filter.GetOutput()

def poly_scale(poly, scale):
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    return transform_filter.GetOutput()

def insert_svol_tomo(svol, tomo, center, merge='sum', edge_softness=3.0, use_antialiasing=True, edge_profile='gaussian', density_threshold=0.01):
    """
    Insert particle model into volume
    
    Args:
        svol: Particle model
        tomo: Target volume
        center: Particle center position
        merge: Merge mode (sum, max, min, alpha_blend)
        edge_softness: Edge softening radius
        edge_profile: Edge transition curve
        density_threshold: Density threshold
    """
    svol_shape = np.array(svol.shape)
    tomo_shape = np.array(tomo.shape)
    center = np.array(center, dtype=int)
    
    half_size = svol_shape // 2
    
    t_min = np.maximum(0, center - half_size)
    t_max = np.minimum(tomo_shape, center + half_size + (svol_shape % 2))
    
    s_min = half_size - (center - t_min)
    s_max = svol_shape - (half_size - (t_max - center - (svol_shape % 2)))
    
    if np.any(s_min >= s_max) or np.any(t_min >= t_max):
        print(f"Warning: Invalid insertion range - svol region: {s_min}-{s_max}, tomo region: {t_min}-{t_max}")
        return center
    
    svol_part = svol[s_min[0]:s_max[0], s_min[1]:s_max[1], s_min[2]:s_max[2]]
    tomo_part = tomo[t_min[0]:t_max[0], t_min[1]:t_max[1], t_min[2]:t_max[2]]
    
    if svol_part.shape != tomo_part.shape:
        print(f"Warning: Shape mismatch - svol part: {svol_part.shape}, tomo part: {tomo_part.shape}")
        common_shape = tuple(min(s, t) for s, t in zip(svol_part.shape, tomo_part.shape))
        svol_part = svol_part[:common_shape[0], :common_shape[1], :common_shape[2]]
        tomo_part = tomo_part[:common_shape[0], :common_shape[1], :common_shape[2]]
    
    density_mask = svol_part > density_threshold
    
    if not np.any(density_mask):
        print(f"Warning: Inserted particle has no density values above threshold ({density_threshold})")
        return center
        
    if edge_softness > 0:
        center_local = np.array(svol_part.shape) // 2
        y, x, z = np.ogrid[:svol_part.shape[0], :svol_part.shape[1], :svol_part.shape[2]]
        dist = np.sqrt((x - center_local[1])**2 + (y - center_local[0])**2 + (z - center_local[2])**2)
        
        max_dist = np.max(dist * density_mask) if np.any(density_mask) else 0
        inner_radius = max(0, max_dist - edge_softness)
        
        if edge_profile == 'gaussian':
            sigma = edge_softness / 2.0
            alpha = np.exp(-((dist - inner_radius) / sigma)**2 / 2.0)
            alpha = np.clip(alpha, 0, 1)
            alpha[dist <= inner_radius] = 1.0
        elif edge_profile == 'linear':
            alpha = 1.0 - (dist - inner_radius) / edge_softness
            alpha = np.clip(alpha, 0, 1)
            alpha[dist <= inner_radius] = 1.0
        elif edge_profile == 'quadratic':
            t = 1.0 - (dist - inner_radius) / edge_softness
            t = np.clip(t, 0, 1)
            alpha = t * t * (3 - 2 * t)
            alpha[dist <= inner_radius] = 1.0
        else:
            alpha = np.ones_like(dist)
            
        alpha = alpha * density_mask
    else:
        alpha = density_mask.astype(float)
    
    if use_antialiasing and edge_softness > 0:
        alpha = gaussian_filter(alpha, sigma=0.5)
        alpha = alpha * (svol_part > 0)
    
    if merge == 'alpha_blend':
        blended = svol_part * alpha + tomo_part * (1 - alpha)
        tomo[t_min[0]:t_max[0], t_min[1]:t_max[1], t_min[2]:t_max[2]] = blended
    elif merge == 'sum':
        tomo[t_min[0]:t_max[0], t_min[1]:t_max[1], t_min[2]:t_max[2]] += svol_part * alpha
    elif merge == 'max':
        weighted_svol = svol_part * alpha
        max_vals = np.copy(tomo_part)
        
        update_mask = alpha > 0
        if np.any(update_mask):
            max_vals[update_mask] = np.maximum(tomo_part[update_mask], weighted_svol[update_mask])
            
        edge_mask = (alpha > 0) & (alpha < 1)
        if np.any(edge_mask):
            edge_blend = alpha[edge_mask] * svol_part[edge_mask] + (1 - alpha[edge_mask]) * tomo_part[edge_mask]
            max_vals[edge_mask] = edge_blend
            
        tomo[t_min[0]:t_max[0], t_min[1]:t_max[1], t_min[2]:t_max[2]] = max_vals
    elif merge == 'min':
        weighted_svol = svol_part * alpha
        min_vals = np.copy(tomo_part)
        
        update_mask = alpha > 0
        if np.any(update_mask):
            min_vals[update_mask] = np.minimum(tomo_part[update_mask], weighted_svol[update_mask])
            
        edge_mask = (alpha > 0) & (alpha < 1)
        if np.any(edge_mask):
            edge_blend = alpha[edge_mask] * svol_part[edge_mask] + (1 - alpha[edge_mask]) * tomo_part[edge_mask]
            min_vals[edge_mask] = edge_blend
            
        tomo[t_min[0]:t_max[0], t_min[1]:t_max[1], t_min[2]:t_max[2]] = min_vals
    
    return center

def generate_ice_layer(shape, voxel_size, ice_thickness_mean=100, ice_thickness_std=20, 
                      noise_scale=10, noise_amplitude=5, density_fluctuation=0.05):
    """
    Generate simulated ice layer
    
    Args:
        shape: Shape of the layer (x,y)
        voxel_size: Size of each voxel in Angstroms
        ice_thickness_mean: Mean ice thickness in Angstroms
        ice_thickness_std: Standard deviation of ice thickness
        noise_scale: Scale of Perlin noise
        noise_amplitude: Amplitude of Perlin noise
        density_fluctuation: Standard deviation of density fluctuations
    
    Returns:
        3D array representing the ice layer
    """
    x_size, y_size = shape[0], shape[1]
    
    ice_thickness = np.random.lognormal(np.log(ice_thickness_mean), ice_thickness_std/ice_thickness_mean)
    ice_thickness_voxels = int(ice_thickness / voxel_size)
    
    ice_layer = np.zeros((x_size, y_size, ice_thickness_voxels), dtype=np.float32)
    
    perlin_grid = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            perlin_grid[i,j] = noise.pnoise2(i/noise_scale, j/noise_scale, octaves=4) * noise_amplitude
    
    perlin_grid = gaussian_filter(perlin_grid, sigma=2.0)
    
    for i in range(x_size):
        for j in range(y_size):
            local_thickness = ice_thickness_voxels - int(perlin_grid[i,j])
            local_thickness = max(ice_thickness_voxels // 2, local_thickness)
            
            ice_layer[i, j, :local_thickness] = 1.0
            
            fluctuations = np.random.normal(0, density_fluctuation, local_thickness)
            ice_layer[i, j, :local_thickness] += fluctuations
    
    return ice_layer

def get_scale_adaptive_parameters(scale_factor, particle_size, pixel_size):
    """
    Return adaptive parameters based on scale factor
    
    Args:
        scale_factor: Scale factor (0-1)
        particle_size: Base particle size
        pixel_size: Pixel size
        
    Returns:
        Dictionary of adaptive parameters
    """
    overlap_threshold = 0.4 - 0.3 * scale_factor
    
    placement_density = 0.7 + 0.5 * scale_factor
    
    detail_level = scale_factor
    
    collision_strictness = 0.5 + 0.5 * scale_factor
    
    mesh_reduction = 0.7 - 0.7 * scale_factor
    
    return {
        'overlap_threshold': overlap_threshold,
        'placement_density': placement_density,
        'detail_level': detail_level,
        'collision_strictness': collision_strictness,
        'mesh_reduction': mesh_reduction
    }

def place_particles(volume_shape, n_particles, placement_strategy='uniform', 
                   gaussian_params=None, cluster_params=None, voi_mask=None, 
                   particle_size=None, avoid_overlap=True, overlap_threshold=0.2,
                   fixed_z=None, interaction_strength=0, exclusion_mask=None, 
                   local_density_map=None, interface_params=None, filament_params=None,
                   scale_factor=0.5):
    """
    Place particles and return their positions
    
    Args:
        volume_shape: Volume shape
        n_particles: Number of particles
        placement_strategy: Strategy type
        particle_size: Particle radius for overlap detection
        avoid_overlap: Avoid particle overlap
        overlap_threshold: Allowed overlap ratio
        fixed_z: Fixed Z coordinate for 2D simulation
        scale_factor: Scale factor (0-1)
    
    Returns:
        positions: List of particle positions
    """
    positions = []
    volume_shape = np.array(volume_shape)
    
    scale_params = get_scale_adaptive_parameters(scale_factor, particle_size, 1.0)
    
    adjusted_overlap_threshold = overlap_threshold * (1 + scale_params['overlap_threshold'])
    collision_strictness = scale_params['collision_strictness']
    
    adjusted_n_particles = int(n_particles * scale_params['placement_density'])
    if adjusted_n_particles != n_particles:
        print(f"Scale adaptation: Adjusting particle count from {n_particles} to {adjusted_n_particles}")
        n_particles = adjusted_n_particles
    
    z_value = fixed_z if fixed_z is not None else None
    
    if exclusion_mask is None and voi_mask is not None:
        exclusion_mask = ~voi_mask
    elif exclusion_mask is None:
        exclusion_mask = np.zeros(volume_shape, dtype=bool)
    
    is_2d_mask = len(exclusion_mask.shape) == 2 or exclusion_mask.shape[2] == 1
    
    if local_density_map is not None and placement_strategy != 'local_density':
        print("Local density map is only used in 'local_density' strategy")
    
    if placement_strategy == 'uniform':
        for _ in range(n_particles * 2):
            if len(positions) >= n_particles:
                break
                
            if z_value is not None:
                pos_xy = np.random.uniform(0, 1, 2) * (volume_shape[:2] - 2*particle_size) + particle_size
                pos = np.array([pos_xy[0], pos_xy[1], z_value], dtype=int)
            else:
                pos = np.random.uniform(0, 1, 3) * (volume_shape - 2*particle_size) + particle_size
                pos = pos.astype(int)
                
            if is_2d_mask:
                if exclusion_mask[pos[0], pos[1]]:
                    continue
            else:
                if exclusion_mask[tuple(np.minimum(volume_shape-1, pos).astype(int))]:
                    continue
                
            if avoid_overlap and positions:
                if z_value is not None:
                    collision_distance = particle_size * (2 - adjusted_overlap_threshold) * collision_strictness
                    distances_xy = [np.linalg.norm(np.array(p[:2]) - pos[:2]) for p in positions]
                    if min(distances_xy, default=float('inf')) < collision_distance:
                        continue
                else:
                    distances = [np.linalg.norm(np.array(p) - pos) for p in positions]
                    if min(distances, default=float('inf')) < particle_size * (2 - adjusted_overlap_threshold):
                        continue
            
            if interaction_strength != 0 and positions:
                interaction_force = 0
                for p in positions:
                    dist = np.linalg.norm(np.array(p) - pos)
                    interaction_force += interaction_strength / (dist + 1e-6)
                
                if interaction_strength < 0 and interaction_force < interaction_strength * 2:
                    continue
                elif interaction_strength > 0 and interaction_force < interaction_strength * 0.5:
                    if np.random.random() > 0.3:
                        continue
            
            positions.append(pos)
    
    if len(positions) < n_particles:
        print(f"Warning: Could only place {len(positions)}/{n_particles} particles")
    
    return positions[:n_particles]

def sample_orientations(n, method='uniform', preferred_direction=None, concentration=None, restricted_angle=None):
    """
    Sample particle orientations
    
    Args:
        n: Number of orientations
        method: Orientation method (uniform, preferred_axis, limited_tilt, etc.)
        preferred_direction: Preferred direction vector
        concentration: Concentration parameter
        restricted_angle: Maximum tilt angle
    
    Returns:
        orientations: List of orientation quaternions
    """
    orientations = []
    
    if preferred_direction is None:
        preferred_direction = np.array([0, 0, 1])
    else:
        preferred_direction = np.array(preferred_direction)
        preferred_direction = preferred_direction / np.linalg.norm(preferred_direction)
    
    if concentration is None:
        concentration = 10.0

    if restricted_angle is None:
        restricted_angle = np.pi/6
    
    if method == 'uniform':
        for _ in range(n):
            rot = Rotation.random()
            orientations.append(rot.as_quat())
    
    elif method == 'preferred_axis':
        for _ in range(n):
            z = 1 - (1 + np.random.exponential(1/concentration)) / (1 + concentration)
            phi = np.random.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(phi)
            y = np.sqrt(1 - z**2) * np.sin(phi)
            
            direction = np.array([x, y, z])
            
            if not np.allclose(preferred_direction, [0, 0, 1]):
                v = np.cross([0, 0, 1], preferred_direction)
                s = np.linalg.norm(v)
                if s > 1e-10:
                    c = np.dot([0, 0, 1], preferred_direction)
                    v_cross = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2)
                    direction = rotation_matrix.dot(direction)
            
            angle = np.random.uniform(0, 2*np.pi)
            rot = Rotation.from_rotvec(angle * direction)
            orientations.append(rot.as_quat())
    
    elif method == 'limited_tilt':
        for _ in range(n):
            theta = np.random.uniform(0, restricted_angle)
            phi = np.random.uniform(0, 2*np.pi)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            direction = np.array([x, y, z])
            
            if not np.allclose(preferred_direction, [0, 0, 1]):
                v = np.cross([0, 0, 1], preferred_direction)
                s = np.linalg.norm(v)
                if s > 1e-10:
                    c = np.dot([0, 0, 1], preferred_direction)
                    v_cross = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2)
                    direction = rotation_matrix.dot(direction)
            
            angle = np.random.uniform(0, 2*np.pi)
            rot = Rotation.from_rotvec(angle * direction)
            orientations.append(rot.as_quat())
    
    elif method == 'equatorial':
        for _ in range(n):
            theta = np.pi/2 + np.random.normal(0, np.pi/12)
            phi = np.random.uniform(0, 2*np.pi)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            direction = np.array([x, y, z])
            
            if not np.allclose(preferred_direction, [0, 0, 1]):
                v = np.cross([0, 0, 1], preferred_direction)
                s = np.linalg.norm(v)
                if s > 1e-10:
                    c = np.dot([0, 0, 1], preferred_direction)
                    v_cross = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2)
                    direction = rotation_matrix.dot(direction)
            
            angle = np.random.uniform(0, 2*np.pi)
            rot = Rotation.from_rotvec(angle * direction)
            orientations.append(rot.as_quat())
    
    elif method == 'bimodal':
        second_direction = -preferred_direction
        
        for _ in range(n):
            if np.random.random() < 0.5:
                current_direction = preferred_direction
            else:
                current_direction = second_direction
            
            z = 1 - (1 + np.random.exponential(1/concentration)) / (1 + concentration)
            phi = np.random.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(phi)
            y = np.sqrt(1 - z**2) * np.sin(phi)
            
            direction = np.array([x, y, z])
            
            if not np.allclose(current_direction, [0, 0, 1]):
                v = np.cross([0, 0, 1], current_direction)
                s = np.linalg.norm(v)
                if s > 1e-10:
                    c = np.dot([0, 0, 1], current_direction)
                    v_cross = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    rotation_matrix = np.eye(3) + v_cross + v_cross.dot(v_cross) * (1 - c) / (s ** 2)
                    direction = rotation_matrix.dot(direction)
            
            angle = np.random.uniform(0, 2*np.pi)
            rot = Rotation.from_rotvec(angle * direction)
            orientations.append(rot.as_quat())
    
    else:
        raise ValueError(f"Orientation sampling method '{method}' not recognized")
    
    return orientations

def compute_safe_freqs(n_pixels, psize):
    """Calculate safe frequency grid"""
    freq_pix_1d = np.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = np.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe)
    rho = np.sqrt(x**2 + y**2)
    angles_rad = np.arctan2(y, x)
    freq_mag_2d = rho / psize
    return freq_mag_2d, angles_rad

def compute_ctf(s, a, dfu, dfv, dfang_deg, kv, cs, w, phase=0, bf=0):
    """
    Calculate Contrast Transfer Function (CTF)
    
    Args:
        s: Frequency magnitude
        a: Frequency angle
        dfu, dfv: U,V direction defocus
        dfang_deg: Astigmatism angle
        kv: Acceleration voltage
        cs: Spherical aberration coefficient
        w: Amplitude contrast
        phase: Phase shift
        bf: B-factor
    """
    s = s[None, ...]
    a = a[None, ...]
    kv = kv * 1e3
    cs = cs * 1e7
    
    lamb = 12.2643247 / np.sqrt(kv * (1.0 + kv * 0.978466e-6))
    
    dfang_deg = np.deg2rad(dfang_deg)
    def_avg = -(dfu + dfv) * 0.5
    def_dev = -(dfu - dfv) * 0.5
    
    k1 = np.pi / 2.0 * 2 * lamb
    k2 = np.pi / 2.0 * cs * lamb**3
    k3 = np.sqrt(1 - w**2)
    k4 = bf / 4.0
    k5 = np.deg2rad(phase)
    
    s_2 = s**2
    s_4 = s_2**2
    
    dZ = def_avg + def_dev * (np.cos(2 * (a - dfang_deg)))
    
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    
    ctf = -(k3 * np.sin(gamma) - w * np.cos(gamma))
    
    if bf != 0:
        ctf *= np.exp(-k4 * s_2)
        
    return ctf

def fft2_center(img):
    """Centered 2D Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(img), axes=(-2, -1))

def ifft2_center(img):
    """Centered 2D inverse Fourier transform"""
    return np.fft.ifft2(np.fft.ifftshift(img, axes=(-2, -1)))

def apply_ctf(image, pixel_size=1.0, voltage=300.0, defocus=-15000.0, cs=2.7, amplitude_contrast=0.07, 
             b_factor=8000.0, phase_shift=0.0, astigmatism_angle=0.0, astigmatism_magnitude=0.0):
    """
    Apply CTF to image using frequency domain method
    
    Args:
        image: Input projection
        pixel_size: Pixel size
        voltage: Acceleration voltage
        defocus: Defocus value, negative for underfocus
        cs: Spherical aberration coefficient
        amplitude_contrast: Amplitude contrast
        b_factor: B-factor
        phase_shift: Phase shift
        astigmatism_angle: Astigmatism angle
        astigmatism_magnitude: Astigmatism magnitude
    """
    ny, nx = image.shape
    
    freq_mag_2d, angles_rad = compute_safe_freqs(nx, pixel_size)
    
    defocus_u = defocus - astigmatism_magnitude / 2
    defocus_v = defocus + astigmatism_magnitude / 2
    
    dfu = np.array([[[defocus_u]]])
    dfv = np.array([[[defocus_v]]])
    dfang_deg = np.array([[[astigmatism_angle]]])
    volt = np.array([[[voltage]]])
    cs_arr = np.array([[[cs]]])
    w = np.array([[[amplitude_contrast]]])
    phase = np.array([[[phase_shift]]])
    
    ctf = compute_ctf(freq_mag_2d, angles_rad, dfu, dfv, dfang_deg, volt, cs_arr, w, phase, b_factor)
    
    img_ft = fft2_center(image)
    img_ft_ctf = img_ft * ctf[0]
    img_ctf = np.real(ifft2_center(img_ft_ctf))
    
    img_ctf = (img_ctf - np.min(img_ctf)) / (np.max(img_ctf) - np.min(img_ctf))
    
    return img_ctf

def project_volume(volume, projection_axis=2, threshold=0.005):
    """
    Project volume data
    
    Args:
        volume: Input volume
        projection_axis: Projection axis (0=X, 1=Y, 2=Z)
        threshold: Threshold for clearing low values
    """
    if projection_axis == 2:
        projection = np.sum(volume, axis=2)
    elif projection_axis == 1:
        projection = np.sum(volume, axis=1)
    elif projection_axis == 0:
        projection = np.sum(volume, axis=0)
    
    if np.max(projection) > 0:
        projection = projection / np.max(projection)
    
    projection[projection < threshold] = 0
    
    return projection

def add_noise(image, snr=0.1, readout_noise=0.01, shot_noise=True, dose=40.0):
    image = np.nan_to_num(image)
    image = np.clip(image, 0, None)
    
    noise_level = np.sqrt(np.mean(image**2) / snr)
    
    if shot_noise:
        image_for_poisson = image * dose
        image_for_poisson = np.clip(image_for_poisson, 0, None)
        image_electrons = np.random.poisson(image_for_poisson)
        image = image_electrons / max(dose, 1e-10)
    
    gaussian_noise = np.random.normal(0, noise_level, image.shape)
    
    if readout_noise > 0:
        readout = np.random.normal(0, readout_noise, image.shape)
        gaussian_noise += readout
    
    noisy_image = image + gaussian_noise
    return noisy_image

def rotate_volume(volume, quaternion, padding_mode='constant', resize=True, safety_factor=1.5):
    """
    Rotate 3D volume using quaternion
    
    Args:
        volume: Input volume
        quaternion: Rotation quaternion
        padding_mode: Padding mode
        resize: Auto-resize to fit rotated volume
        safety_factor: Safety factor for new volume size
    
    Returns:
        Rotated volume
    """
    volume = volume.astype(np.float32)
    
    rot = Rotation.from_quat(quaternion)
    rot_matrix = rot.as_matrix()
    
    orig_shape = np.array(volume.shape)
    orig_center = orig_shape / 2
    
    if resize:
        diagonal = np.sqrt(np.sum(np.array(volume.shape)**2)) * safety_factor
        
        new_shape = np.array([diagonal, diagonal, diagonal], dtype=int)
        new_shape += (1 - new_shape % 2)
        
        new_shape = np.maximum(new_shape, orig_shape)
        
        if np.any(new_shape > orig_shape):
            new_volume = np.zeros(new_shape, dtype=volume.dtype)
            
            offset = ((new_shape - orig_shape) / 2).astype(int)
            
            sl_orig = tuple(slice(None) for _ in range(len(orig_shape)))
            sl_new = tuple(slice(offset[i], offset[i] + orig_shape[i]) for i in range(len(orig_shape)))
            new_volume[sl_new] = volume[sl_orig]
            
            volume = new_volume
            center = new_shape / 2
        else:
            center = orig_center
    else:
        center = orig_center
    
    z, y, x = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    x = x - center[2]
    y = y - center[1]
    z = z - center[0]
    
    coords = np.zeros((3, *volume.shape), dtype=np.float32)
    coords[0, :, :, :] = z
    coords[1, :, :, :] = y
    coords[2, :, :, :] = x
    
    inv_rot_matrix = rot_matrix.T
    
    coords_rot = np.zeros_like(coords)
    for i in range(3):
        for j in range(3):
            coords_rot[i] += inv_rot_matrix[i, j] * coords[j]
    
    coords_rot[0] += center[0]
    coords_rot[1] += center[1]
    coords_rot[2] += center[2]
    
    rotated = ndimage_map_coordinates(volume, coords_rot, order=1, mode=padding_mode)
    
    epsilon = 1e-6
    rotated[np.abs(rotated) < epsilon] = 0
    
    return rotated

def generate_cryoem_micrograph(particles_list, num_particles, pixel_size, volume_shape, 
                              output_dir, placement_strategy='uniform',
                              orientation_method='uniform', ctf_params=None, 
                              add_ice=True, ice_params=None,
                              particle_interaction_strength=0,
                              interface_params=None,
                              filament_params=None,
                              edge_softness=5.0,
                              edge_profile='gaussian',
                              use_antialiasing=True,
                              blend_mode='alpha_blend',
                              density_threshold=0.01,
                              projection_threshold=0.005,
                              diagonal_factor=1.2,
                              rotation_safety_factor=1.5):
    """
    Generate simulated cryo-EM micrograph
    
    Args:
        particles_list: List of particle MRC files
        num_particles: Number of particles
        pixel_size: Pixel size
        volume_shape: Volume shape
        output_dir: Output directory
        placement_strategy: Particle placement strategy
        orientation_method: Orientation sampling method
        ctf_params: CTF parameters
        add_ice: Add ice layer
        density_threshold: Threshold for particle insertion
        diagonal_factor: Controls particle density
        rotation_safety_factor: Prevents particle truncation
    
    Returns:
        projection: Projection image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    volume = np.zeros(volume_shape, dtype=np.float32)
    print(f"Creating volume, shape: {volume.shape}")
    
    total_volume_size = volume_shape[0] * pixel_size
    particle_sizes = []
    particle_models = []
    
    for particle_path in particles_list:
        model = load_mrc(particle_path)
        particle_models.append(model)
        size = max(model.shape) * pixel_size / 2
        particle_sizes.append(size)
    
    avg_particle_size = np.mean(particle_sizes)
    
    scale_factor = detect_scale(avg_particle_size / pixel_size, volume_shape, pixel_size)
    
    adjusted_num_particles = num_particles
    
    if diagonal_factor > 1.1:
        density_compensation = min(diagonal_factor**2, 2.0)
        adjusted_num_particles = int(num_particles * density_compensation)
    
    particles_per_type = [adjusted_num_particles // len(particles_list) + (1 if i < adjusted_num_particles % len(particles_list) else 0) 
                         for i in range(len(particles_list))]
    
    all_positions = []
    all_orientations = []
    
    fixed_z = volume_shape[2] // 2
    print(f"Placing all particles at z={fixed_z} plane")
    
    occupied_mask = np.zeros(volume.shape[:2], dtype=bool)
    
    scale_params = get_scale_adaptive_parameters(scale_factor, avg_particle_size, pixel_size)
    
    print(f"Particle edge processing parameters:")
    print(f"  - Softening radius: {edge_softness} pixels")
    print(f"  - Edge profile: {edge_profile}")
    print(f"  - Antialiasing: {'enabled' if use_antialiasing else 'disabled'}")
    print(f"  - Blend mode: {blend_mode}")
    print(f"  - Density threshold: {density_threshold}")
    print(f"  - Diagonal factor: {diagonal_factor} (controls particle density)")
    print(f"  - Rotation safety factor: {rotation_safety_factor} (prevents truncation)")
    
    for i, (model, count) in enumerate(zip(particle_models, particles_per_type)):
        print(f"Processing particle type {i+1}/{len(particle_models)}: Placing {count} particles")
        print(f"Particle shape: {model.shape}, Particle value range: [{np.min(model)}, {np.max(model)}]")
        
        if not np.issubdtype(model.dtype, np.floating):
            model = model.astype(np.float32)
            print(f"  Converting particle model to float type")
        
        radius = int(np.ceil(particle_sizes[i]))
        
        diagonal_radius = int(np.ceil(diagonal_factor * radius))
        
        overlap_threshold = min(0.5, scale_params['overlap_threshold'] * 1.5)
        
        print(f"  Base particle radius: {radius} pixels, Rotation safety radius: {diagonal_radius} pixels")
        print(f"  Overlap threshold: {overlap_threshold} (0-1, higher allows more overlap)")
        
        exclusion_mask = occupied_mask.copy()
        
        positions = place_particles(
            volume_shape=volume_shape,
            n_particles=count,
            placement_strategy=placement_strategy,
            particle_size=diagonal_radius,
            avoid_overlap=True,
            overlap_threshold=overlap_threshold,
            fixed_z=fixed_z,
            exclusion_mask=exclusion_mask,
            interaction_strength=particle_interaction_strength,
            interface_params=interface_params,
            filament_params=filament_params,
            scale_factor=scale_factor
        )
        
        print(f"Successfully placed {len(positions)} particle positions")
        
        for pos in positions:
            x, y = pos[0], pos[1]
            x_min = max(0, x - diagonal_radius)
            x_max = min(volume_shape[0], x + diagonal_radius)
            y_min = max(0, y - diagonal_radius)
            y_max = min(volume_shape[1], y + diagonal_radius)
            
            xx, yy = np.ogrid[x_min:x_max, y_min:y_max]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            occupied_mask[x_min:x_max, y_min:y_max] |= (dist <= diagonal_radius)
        
        orientations = sample_orientations(
            len(positions), 
            method=orientation_method, 
            preferred_direction=[0, 0, 1],
            concentration=10.0
        )
        
        detail_level = scale_params['detail_level']
        
        for pos_idx, (pos, orient) in enumerate(zip(positions, orientations)):
            particle_vol = np.copy(model)
            
            if np.max(particle_vol) - np.min(particle_vol) > 0:
                particle_vol = (particle_vol - np.min(particle_vol)) / (np.max(particle_vol) - np.min(particle_vol))
            else:
                print(f"  Warning: Particle {pos_idx+1} has no dynamic range")
                continue
            
            vtk_quat = [orient[3], orient[0], orient[1], orient[2]]
            print(f"  Applying rotation: quaternion={vtk_quat}")
            
            try:
                particle_vol = rotate_volume(
                    particle_vol, 
                    orient, 
                    padding_mode='constant', 
                    resize=True,
                    safety_factor=rotation_safety_factor
                )
                
                particle_vol[particle_vol < density_threshold] = 0
                
                particle_vol = particle_vol * 0.5
                
                print(f"  Placing particle {pos_idx+1}/{len(positions)}: position={pos}")
                
                nonzero_before = np.count_nonzero(volume)
                
                if pos_idx == 0:
                    rotated_dir = os.path.join(output_dir, 'rotated_particles')
                    os.makedirs(rotated_dir, exist_ok=True)
                    rotated_path = os.path.join(rotated_dir, f'rotated_particle_{i}.mrc')
                    save_volume_to_mrc(particle_vol, rotated_path, spacing=pixel_size)
                    print(f"  Saved rotated particle: {rotated_path}")
                
                insert_svol_tomo(
                    particle_vol, 
                    volume, 
                    pos, 
                    merge=blend_mode,
                    edge_softness=edge_softness,
                    use_antialiasing=use_antialiasing,
                    edge_profile=edge_profile,
                    density_threshold=density_threshold
                )
                
                nonzero_after = np.count_nonzero(volume)
                if nonzero_after > nonzero_before:
                    print(f"  Success: Added particle, non-zero elements increased: {nonzero_after - nonzero_before}")
                    all_positions.append(pos)
                    all_orientations.append(orient)
                else:
                    print(f"  Warning: Particle may not have been added successfully")
            except Exception as e:
                print(f"  Error placing particle: {e}")
                print(f"  Skipping this particle and continuing")
                continue
    
    particle_volume_path = os.path.join(output_dir, 'particle_volume.mrc')
    save_volume_to_mrc(volume, particle_volume_path, spacing=pixel_size)
    print(f"Saved particle volume: {particle_volume_path}")
    
    if add_ice:
        if ice_params is None:
            ice_layer = generate_ice_layer(volume_shape[:2], pixel_size)
        else:
            ice_layer = generate_ice_layer(volume_shape[:2], pixel_size, **ice_params)
        
        for i in range(volume_shape[0]):
            for j in range(volume_shape[1]):
                ice_values = ice_layer[i, j]
                volume[i, j] = np.maximum(volume[i, j], ice_values[:volume_shape[2]])
    
    print(f"Volume original range: min={np.min(volume)}, max={np.max(volume)}, mean={np.mean(volume)}")

    if np.max(volume) > 0:
        volume = (volume - np.min(volume)) / max(np.max(volume) - np.min(volume), 1e-6)
        volume[volume < density_threshold] = 0
        volume = volume ** 0.5
        print(f"After contrast enhancement: min={np.min(volume)}, max={np.max(volume)}, mean={np.mean(volume)}")
    
    volume_inverted = 1.0 - volume
    background_mask = volume <= density_threshold
    volume_inverted[background_mask] = 1.0
    
    nonzero_count = np.count_nonzero(volume > density_threshold)
    total_voxels = np.prod(volume_shape)
    print(f"Volume info: shape={volume.shape}, non-zero voxels={nonzero_count}, total voxels={total_voxels}")
    print(f"Volume occupancy: {nonzero_count/total_voxels*100:.4f}%")
    print(f"Density range: min={np.min(volume_inverted)}, max={np.max(volume_inverted)}, mean={np.mean(volume_inverted)}")
    
    if nonzero_count > 0:
        if np.max(volume) - np.min(volume) < 0.1:
            print("Warning: Low density contrast, performing contrast enhancement")
            volume = volume * 10
            volume_inverted = 1.0 - volume
            volume_inverted[background_mask] = 1.0
    else:
        print("Error: No non-zero voxels in volume!")
    
    projection = project_volume(volume_inverted, projection_axis=2, threshold=projection_threshold)
    print(f"Projection shape: {projection.shape}")
    
    projection_3d = np.expand_dims(projection, axis=2)
    
    projection_3d[projection_3d < projection_threshold] = 0
    
    save_mrc(projection_3d, os.path.join(output_dir, 'micrograph.mrc'), spacing=pixel_size)
    print(f"Saved micrograph: {os.path.join(output_dir, 'micrograph.mrc')}")

    with open(os.path.join(output_dir, 'particles.txt'), 'w') as f:
        f.write("# Particle positions and orientations\n")
        f.write("# Format: id, pos_x, pos_y, quat_x, quat_y, quat_z, quat_w\n")
        for i, (pos, quat) in enumerate(zip(all_positions, all_orientations)):
            f.write(f"{i}, {pos[0]}, {pos[1]}, {quat[0]}, {quat[1]}, {quat[2]}, {quat[3]}\n")
    
    return projection

def is_pdb_file(file_path):
    """Check if file is in PDB format"""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in ['.pdb', '.ent']

def is_mrc_file(file_path):
    """Check if file is in MRC format"""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in ['.mrc', '.map', '.ccp4']

def get_atom_radius(atom):
    """Get Van der Waals radius of an atom (Angstrom)"""
    radius_dict = {
        'H': 1.2,
        'C': 1.7,
        'N': 1.55,
        'O': 1.52,
        'P': 1.8,
        'S': 1.8,
        'FE': 2.0,
        'ZN': 1.39,
        'MG': 1.73,
        'CA': 2.31,
        'NA': 2.27,
        'CL': 1.75
    }
    
    element = atom.element.strip().upper()
    if element in radius_dict:
        return radius_dict[element]
    return 1.7

def convert_pdb_to_volume(pdb_file, resolution=3.0, box_margin=None, verbose=True, auto_adjust_box=True, background_threshold=0.005):
    """
    Convert PDB file to voxel representation
    
    Args:
        pdb_file: Path to PDB file
        resolution: Target resolution
        box_margin: Margin around molecule
        auto_adjust_box: Auto-adjust box size
        background_threshold: Background threshold
        
    Returns:
        volume: Volume data
        spacing: Voxel spacing
    """
    if verbose:
        print(f"Processing PDB file: {pdb_file}")
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)
    
    atoms = Selection.unfold_entities(structure, 'A')
    
    if len(atoms) == 0:
        raise ValueError(f"No atoms found in PDB file {pdb_file}")
    
    if verbose:
        print(f"PDB contains {len(atoms)} atoms")
    
    coords = np.array([atom.coord for atom in atoms])
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)
    
    molecule_size = np.linalg.norm(max_coord - min_coord)
    
    atom_radii = [get_atom_radius(atom) for atom in atoms]
    max_atom_radius = max(atom_radii)
    
    if box_margin is None:
        if auto_adjust_box:
            box_margin = max_atom_radius * 3.0
            
            box_margin = max(box_margin, resolution * 2)
            box_margin = min(box_margin, molecule_size * 0.2)
        else:
            box_margin = max(max_atom_radius * 3.0, resolution * 2)
    
    if verbose:
        print(f"Using margin: {box_margin:.2f}Å (molecule diameter: {molecule_size:.2f}Å, max atom radius: {max_atom_radius:.2f}Å)")
    
    padded_min_coord = min_coord - box_margin
    padded_max_coord = max_coord + box_margin
    
    spacing = resolution / 2.0
    
    dimensions = np.ceil((padded_max_coord - padded_min_coord) / spacing).astype(int)
    dimensions = np.maximum(dimensions, [1, 1, 1])
    
    dimensions += dimensions % 2
    
    if verbose:
        print(f"Volume dimensions: {dimensions}, voxel size: {spacing}Å")
        print(f"Physical size: {dimensions * spacing} Å")
        
    atoms_would_be_outside = 0
    for atom in atoms:
        atom_coord = atom.coord
        atom_radius = get_atom_radius(atom)
        
        if np.any(atom_coord - atom_radius * 2 < padded_min_coord) or \
           np.any(atom_coord + atom_radius * 2 > padded_max_coord):
            atoms_would_be_outside += 1
            
    if atoms_would_be_outside > 0 and verbose:
        percent = atoms_would_be_outside / len(atoms) * 100
        print(f"Warning: About {atoms_would_be_outside} atoms ({percent:.1f}%) may have electron density edges outside box boundaries.")
        print(f"Consider increasing box_margin or adjusting resolution.")
        if percent > 10 and auto_adjust_box:
            additional_margin = max_atom_radius * 4
            print(f"Automatically increasing margin by {additional_margin:.2f}Å to ensure complete atom density capture")
            return convert_pdb_to_volume(pdb_file, resolution, box_margin + additional_margin, verbose)
    
    volume = np.zeros(dimensions, dtype=np.float32)
    
    atoms_outside_volume = 0
    
    for atom in tqdm(atoms, desc="Creating electron density", disable=not verbose):
        center = np.round((atom.coord - padded_min_coord) / spacing).astype(int)
        
        if not np.all((center >= 0) & (center < dimensions)):
            atoms_outside_volume += 1
            continue
        
        radius = get_atom_radius(atom)
        sigma = radius / spacing / 2.0
        
        r = int(radius / spacing * 2.0)
        x_min = max(0, center[0] - r)
        x_max = min(dimensions[0], center[0] + r + 1)
        y_min = max(0, center[1] - r)
        y_max = min(dimensions[1], center[1] + r + 1)
        z_min = max(0, center[2] - r)
        z_max = min(dimensions[2], center[2] + r + 1)
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                for z in range(z_min, z_max):
                    diff = np.array([x, y, z]) - center
                    dist = np.sqrt(np.sum((diff * spacing)**2))
                    
                    if dist <= radius * 2.0:
                        density = np.exp(-(dist / sigma)**2 / 2)
                        volume[x, y, z] += density
    
    if atoms_outside_volume > 0:
        print(f"Warning: {atoms_outside_volume} atoms ({atoms_outside_volume/len(atoms)*100:.1f}%) are outside volume boundaries. Increase box_margin parameter to solve this issue.")
    
    smooth_sigma = resolution / (2.0 * spacing)
    volume = gaussian_filter(volume, sigma=smooth_sigma)
    
    if np.max(volume) > 0:
        volume = volume / np.max(volume)
    
    volume[volume < background_threshold] = 0
    
    if verbose:
        print(f"Volume created, size {volume.shape}, value range [{np.min(volume)}, {np.max(volume)}]")
        molecule_volume_fraction = np.sum(volume > 0.05) / np.prod(dimensions)
        print(f"Molecule occupies {molecule_volume_fraction*100:.2f}% of volume")
        print(f"Cleared background noise below {background_threshold}")
    
    return volume, spacing

def save_volume_to_mrc(volume, output_path, spacing=1.0, clear_background=True, background_threshold=0.005):
    """
    Save volume data as MRC file
    
    Args:
        volume: Volume data
        output_path: Output file path
        spacing: Voxel spacing
        clear_background: Clear background noise
        background_threshold: Background threshold
    """
    if clear_background:
        volume_copy = volume.copy()
        volume_copy[volume_copy < background_threshold] = 0
    else:
        volume_copy = volume
        
    data_transposed = np.transpose(volume_copy, (2, 1, 0))
    
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data_transposed.astype(np.float32))
        mrc.voxel_size = spacing
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3

def process_input_files(input_files, temp_dir, output_dir, resolution=3.0, box_margin=None, verbose=True, auto_adjust_box=True, background_threshold=0.005):
    """
    Process input files, convert PDB to MRC
    
    Args:
        input_files: List of input files
        temp_dir: Temporary directory path
        output_dir: Output directory path
        resolution: Target resolution
        box_margin: Margin around molecule
        
    Returns:
        processed_files: List of processed files
        converted_files: PDB to MRC mapping
    """
    processed_files = []
    converted_files = {}
    
    os.makedirs(output_dir, exist_ok=True)
    pdb_output_dir = os.path.join(output_dir, "converted_pdbs")
    os.makedirs(pdb_output_dir, exist_ok=True)
    
    for file_path in input_files:
        if is_pdb_file(file_path):
            if verbose:
                print(f"Detected PDB file: {file_path}, will convert to MRC format")
            
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            temp_mrc_path = os.path.join(temp_dir, f"{base_name}.mrc")
            
            final_mrc_path = os.path.join(pdb_output_dir, f"{base_name}.mrc")
            
            try:
                volume, spacing = convert_pdb_to_volume(
                    file_path, 
                    resolution=resolution, 
                    box_margin=box_margin,
                    verbose=verbose,
                    auto_adjust_box=auto_adjust_box,
                    background_threshold=background_threshold
                )
                
                save_volume_to_mrc(
                    volume, 
                    temp_mrc_path, 
                    spacing, 
                    clear_background=True, 
                    background_threshold=background_threshold
                )
                
                shutil.copy2(temp_mrc_path, final_mrc_path)
                
                if verbose:
                    print(f"Converted PDB file to MRC: {temp_mrc_path}")
                    print(f"Converted MRC file also saved at: {final_mrc_path}")
                    print(f"Cleared background noise below {background_threshold}")
                
                processed_files.append(temp_mrc_path)
                converted_files[file_path] = final_mrc_path
                
            except Exception as e:
                print(f"Error processing PDB file {file_path}: {e}")
                continue
                
        else:
            print(f"Warning: Only PDB file format is supported, file {file_path} will be skipped")
    
    return processed_files, converted_files

def main():
    parser = argparse.ArgumentParser(description='CryoEM Simulator')
    parser.add_argument('--input', nargs='+', required=True, help='Input PDB files')
    parser.add_argument('--output', type=str, default='./cryoem_output', help='Output directory')
    parser.add_argument('--num_particles', type=int, default=50, help='Number of particles to place')
    parser.add_argument('--pixel_size', type=float, default=1.0, help='Pixel size in Angstroms')
    parser.add_argument('--volume_shape', type=int, nargs=3, default=[1024, 1024, 128], help='Volume shape (x, y, z)')
    parser.add_argument('--placement', type=str, default='uniform', 
                      choices=['uniform', 'gaussian', 'cluster', 'gradient', 'grid', 'interface', 'filament', 'membrane'], 
                      help='Particle placement strategy')
    parser.add_argument('--orientation', type=str, default='uniform', 
                      choices=['uniform', 'preferred_axis', 'limited_tilt', 'equatorial', 'bimodal'], 
                      help='Orientation sampling method')
    parser.add_argument('--ice', action='store_true', help='Add ice layer')
    parser.add_argument('--resolution', type=float, default=3.0, help='Resolution for PDB to MRC conversion (Angstroms)')
    parser.add_argument('--box_margin', type=float, default=None, help='Box margin for PDB to MRC conversion (Angstroms). If None, automatically determined')
    parser.add_argument('--no_auto_adjust_box', action='store_true', help='Disable automatic box size adjustment')
    parser.add_argument('--background_threshold', type=float, default=0.005, 
                      help='Background threshold for PDB conversion (0-1). Densities below this value will be set to 0.')
    parser.add_argument('--projection_threshold', type=float, default=0.005, 
                      help='Threshold for projection cleaning (0-1). Values below this in projection will be set to 0.')
    parser.add_argument('--diagonal_factor', type=float, default=1.2,
                      help='Diagonal factor for rotation safety (1.0-2.0). Lower values make particles more dense but risk clipping; higher values are safer but more sparse.')
    parser.add_argument('--rotation_safety_factor', type=float, default=1.5,
                      help='Safety factor for particle rotation (1.0-2.0). Higher values prevent clipping but increase memory usage.')
    parser.add_argument('--edge_softness', type=float, default=5.0, help='Edge softness for particle boundaries (pixels)')
    parser.add_argument('--edge_profile', type=str, default='gaussian', 
                      choices=['gaussian', 'linear', 'quadratic'], help='Edge transition profile')
    parser.add_argument('--no_antialiasing', action='store_true', help='Disable antialiasing for particle edges')
    parser.add_argument('--blend_mode', type=str, default='alpha_blend',
                      choices=['alpha_blend', 'sum', 'max', 'min'], help='Particle blending mode')
    parser.add_argument('--density_threshold', type=float, default=0.01, 
                      help='Density threshold for particle insertion (0-1). Only voxels with density above this value are inserted')
    
    args = parser.parse_args()
    print(f"Using volume size: {args.volume_shape}")
    
    temp_dir = tempfile.mkdtemp(prefix="cryoem_")
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        processed_files, converted_files = process_input_files(
            args.input, 
            temp_dir,
            args.output,
            resolution=args.resolution,
            box_margin=args.box_margin,
            verbose=True,
            auto_adjust_box=not args.no_auto_adjust_box,
            background_threshold=args.background_threshold
        )
        
        if not processed_files:
            print("Error: No valid PDB files")
            sys.exit(1)
        
        print(f"Processed file list: {processed_files}")
        print(f"Converted {len(converted_files)} PDB files to MRC format")
        print(f"Cleared noise using background threshold {args.background_threshold}")
        
        ctf_params = {
            'pixel_size': args.pixel_size,
            'voltage': 300.0,
            'defocus': -15000.0,
            'cs': 2.7,
            'amplitude_contrast': 0.07,
            'b_factor': 8000.0,
            'phase_shift': 0.0
        }
        
        edge_params = {
            'edge_softness': args.edge_softness,
            'use_antialiasing': not args.no_antialiasing,
            'edge_profile': args.edge_profile,
            'blend_mode': args.blend_mode,
            'density_threshold': args.density_threshold
        }
        
        print(f"Particle edge processing: softening={args.edge_softness} pixels, antialiasing={not args.no_antialiasing}, edge profile={args.edge_profile}, blend mode={args.blend_mode}")
        print(f"Density threshold: {args.density_threshold} (only insert voxels above this value)")
        print(f"Projection threshold: {args.projection_threshold} (clear background below this value in projection)")
        print(f"Diagonal factor: {args.diagonal_factor} (controls space for rotated particles, smaller is denser)")
        print(f"Rotation safety factor: {args.rotation_safety_factor} (prevents particle truncation during rotation)")
        
        config = {
            'simulation_params': {
                'num_particles': args.num_particles,
                'pixel_size': args.pixel_size,
                'volume_shape': args.volume_shape,
                'placement_strategy': args.placement,
                'orientation_method': args.orientation,
                'add_ice': args.ice,
                'diagonal_factor': args.diagonal_factor,
                'rotation_safety_factor': args.rotation_safety_factor
            },
            'edge_params': edge_params,
            'ctf_params': ctf_params,
            'pdb_conversion_params': {
                'resolution': args.resolution,
                'box_margin': args.box_margin,
                'auto_adjust_box': not args.no_auto_adjust_box,
                'background_threshold': args.background_threshold,
                'projection_threshold': args.projection_threshold
            },
            'input_files': args.input
        }
        
        with open(os.path.join(args.output, "simulation_config.json"), "w") as f:
            import json
            json.dump(config, f, indent=2)
        
        generate_cryoem_micrograph(
            particles_list=processed_files,
            num_particles=args.num_particles,
            pixel_size=args.pixel_size,
            volume_shape=args.volume_shape,
            output_dir=args.output,
            placement_strategy=args.placement,
            orientation_method=args.orientation,
            ctf_params=ctf_params,
            add_ice=args.ice,
            edge_softness=args.edge_softness,
            edge_profile=args.edge_profile,
            use_antialiasing=not args.no_antialiasing,
            blend_mode=args.blend_mode,
            density_threshold=args.density_threshold,
            projection_threshold=args.projection_threshold,
            diagonal_factor=args.diagonal_factor,
            rotation_safety_factor=args.rotation_safety_factor
        )
    
        print(f"Micrograph generation complete. Results saved in {args.output}")
        
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
