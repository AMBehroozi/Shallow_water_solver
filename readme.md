# Shallow Water Solver (FVM-based on Unstructured Meshes)

This project provides a GPU-accelerated, finite volume method (FVM) solver for the two-dimensional shallow water equations on unstructured triangular meshes. Implemented in PyTorch, it simulates dynamic water surface behavior over variable bed topography using explicit time stepping and second-order spatial reconstruction.

## 🔧 Features

- ⚡ GPU-accelerated computations with PyTorch
- 🧩 Supports unstructured triangular meshes
- 🌊 Solves the 2D shallow water equations with varying bed elevation
- 🏔️ Handles bed slope source terms via analytical topography functions
- 📈 Animation and visualization utilities for height and velocity fields
- 🧠 High-order polynomial reconstruction for bed slope derivatives
- 🧱 Built-in wall boundary condition implementation

## 📘 Model Overview

The solver evolves the conservative form of the 2D shallow water equations:
- Mass conservation
- Momentum conservation in x and y
- Source terms account for bed slope effects

## 🧪 Key Components

### 1. Initial Conditions

`create_initial_condition()`  
Creates a discontinuous water surface height based on spatial location to initiate dam-break-like scenarios.

### 2. Bed Topography

`create_bed_topography()`  
Defines the bed using a Gaussian bump function and computes its slope for use in source terms.

### 3. Bed Slope Derivatives

`compute_bed_derivatives()`  
Reconstructs partial derivatives of the bed elevation using a polynomial least-squares approach with neighborhood sampling.

### 4. Core Solver

`TorchMeshSolver`  
Main class that handles:
- Finite volume updates over time
- Numerical flux computation at cell interfaces
- Application of wall boundary conditions
- Source term contributions from bed slopes

### 5. Visualization

`simulate_and_plot_torch()`  
Generates real-time animated plots of:
- Water surface height
- Velocity magnitude
- Bed topography

## 🚀 How to Run

You can execute the solver in **Google Colab** or locally with CUDA support. For a minimal working example:

