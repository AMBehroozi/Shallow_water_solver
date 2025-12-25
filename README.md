# Shallow Water Solver

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMBehroozi/Shallow_water_solver/blob/main/solver_SWE.ipynb)

A GPU-accelerated finite volume method (FVM) solver for the 2D shallow water equations on unstructured triangular meshes, implemented in PyTorch.

## Features

- ⚡ **GPU Acceleration** - Leverages PyTorch for high-performance GPU computing
- 🧩 **Unstructured Meshes** - Supports arbitrary triangular mesh geometries
- 🌊 **Complete Physics** - Solves full 2D shallow water equations with bed topography
- 📈 **High-Order Accuracy** - Second-order spatial reconstruction for improved accuracy
- 🎬 **Visualization** - Built-in animation tools for water height and velocity fields

## Mathematical Model

The solver implements the conservative form of the 2D shallow water equations:

```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
∂(hu)/∂t + ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y = -gh∂b/∂x
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + ½gh²)/∂y = -gh∂b/∂y
```

where:
- `h` = water height
- `u, v` = velocity components
- `g` = gravitational acceleration
- `b` = bed elevation

## Installation

### Requirements
```bash
pip install numpy scipy matplotlib torch gdown tqdm
```

### Quick Start
```python
# Download mesh file
import gdown
file_id = '1zA1DlxYne3YQgyVUpAm7mJkmaROF-Ali'
gdown.download(f'https://drive.google.com/uc?id={file_id}', 'narrowing_channel.ply2', quiet=False)

# Initialize mesh and solver
from solver_SWE import TriMesh, TorchMeshSolver, create_initial_condition, simulate_and_plot_torch

tri_mesh = TriMesh('narrowing_channel', '.ply2', refine=0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set initial conditions
initial_height = create_initial_condition(tri_mesh.cells, device)
solver = TorchMeshSolver(tri_mesh, device=device)
solver.init_height(initial_height)

# Run simulation
ani = simulate_and_plot_torch(
    solver,
    target_time=2.0,
    dt=0.005,
    fps=30,
    speed_up=10,
    norm_h=Normalize(0, 5),
    norm_v=Normalize(0, 3)
)
```

## Key Components

### Mesh Handling (`TriMesh`)
- Reads `.ply2` mesh format
- Computes geometric properties (areas, normals, face lengths)
- Handles boundary conditions (wall boundaries)

### Initial Conditions
- `create_initial_condition()` - Dam-break scenario with discontinuous water height
- `create_bed_topography()` - Gaussian bump bed elevation

### Bed Slope Computation
- `compute_bed_derivatives()` - High-order polynomial reconstruction
- Uses least-squares fitting with extended neighborhood sampling
- Supports both quadratic and linear fallback methods

### Numerical Solver (`TorchMeshSolver`)
- Explicit time-stepping scheme
- Rusanov (local Lax-Friedrichs) flux computation
- Wall boundary conditions with velocity reflection
- Source terms for bed slope effects

### Visualization
- `simulate_and_plot_torch()` - Real-time animation generation
- Displays water height, velocity magnitude, and bed topography
- Exports to HTML5 video format

## Technical Details

**Numerical Method**: Finite Volume Method (FVM)  
**Flux Scheme**: Rusanov (Local Lax-Friedrichs)  
**Time Integration**: Explicit Euler  
**Spatial Reconstruction**: Second-order polynomial  
**Boundary Conditions**: Reflective wall boundaries  

## Example Output

The solver generates animations showing:
1. Water surface height evolution
2. Velocity magnitude distribution
3. Static bed topography

Perfect for simulating:
- Dam-break flows
- Flood propagation
- Channel hydraulics
- Coastal inundation

## License

MIT License - See LICENSE file for details

## Citation

If you use this solver in your research, please cite:
```bibtex
@software{shallow_water_solver,
  author = {AMBehroozi},
  title = {Shallow Water Solver: GPU-Accelerated FVM for Unstructured Meshes},
  year = {2024},
  url = {https://github.com/AMBehroozi/Shallow_water_solver}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
