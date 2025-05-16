# CryoEM Simulator

A simulator for generating cryo-electron microscopy (CryoEM) micrographs with physically accurate features.

## Key Features

- **Structure Processing**: Converts atomic coordinates to electron density volumes with controllable parameters
- **Multi-Scale Volume Modeling**: Adjusts mesh complexity and placement parameters based on particle size and scale
- **Class-Specific Distribution Modeling**: Implements various particle placement strategies
- **Orientation Sampling**: Supports multiple sampling methods
- **Ice-Layer Modeling**: Simulates vitrified ice with realistic thickness variations and density fluctuations
- **Projection and CTF Application**: Computes projections and applies contrast transfer function with configurable parameters

## Installation

### Requirements

```
numpy
mrcfile
scipy
matplotlib
scikit-image
tqdm
biopython
vtk
```

Install dependencies:

```bash
pip install numpy mrcfile scipy matplotlib scikit-image tqdm biopython vtk
```

## Usage

Basic usage:

```bash
python cryoem/cryoem_simulator.py --input path/to/your.pdb --output ./output_dir
```

### Key Parameters

- `--input`: Input PDB file(s) [required]
- `--output`: Output directory (default: './cryoem_output')
- `--num_particles`: Number of particles to place (default: 50)
- `--volume_shape`: Volume dimensions (default: 1024 1024 128)
- `--pixel_size`: Pixel size in Angstroms (default: 1.0)
- `--resolution`: Target resolution for PDB conversion (default: 3.0)
- `--placement`: Particle placement strategy (default: 'uniform', choices: uniform, gaussian, cluster, gradient, grid, interface, filament, membrane)
- `--orientation`: Orientation sampling method (default: 'uniform', choices: uniform, preferred_axis, limited_tilt, equatorial, bimodal)
- `--ice`: Add ice layer with realistic properties (flag)
- `--edge_softness`: Edge softening for particle boundaries (default: 5.0)
- `--edge_profile`: Edge transition profile (default: 'gaussian', choices: gaussian, linear, quadratic)
- `--blend_mode`: Particle blending mode (default: 'alpha_blend', choices: alpha_blend, sum, max, min)
- `--density_threshold`: Threshold for particle insertion (default: 0.01)
- `--background_threshold`: Background threshold for conversion (default: 0.005)
- `--projection_threshold`: Threshold for projection cleaning (default: 0.005)
- `--diagonal_factor`: Controls particle spacing (default: 1.2)
- `--rotation_safety_factor`: Prevents truncation during rotation (default: 1.5)

## Example

Generate a standard micrograph:

```bash
python cryoem_simulator.py --input protein.pdb --output ./results
```

## Output Files

The simulator produces:
- Converted PDB volumes (MRC format)
- Particle volume with all placed particles
- Final micrograph (MRC format)
- Particle positions and orientations text file
- Configuration JSON file with all parameters