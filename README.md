# Lid-Driven Cavity Flow Solver (2D)

High-resolution finite-difference solver for the classical 2D incompressible
lid-driven cavity problem, implemented using the streamfunction–vorticity formulation.

![Showcase](figures/lid_driven_cavity_showcase_Re1000.png)

## Features
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport
- Red–Black SOR for the streamfunction Poisson equation
- Physically consistent coordinate system
- Instant post-processing from saved solutions (`.npz`)

## Usage

Run a simulation:
```bash
python lid_driven_cavity_fdm.py
```
Generate plots from a saved solution:
```bash
python plot_from_npz.py
```
## Output

    Flow fields (ψ, ω, u, v, p) saved as NumPy archives (.npz)

    Publication-quality diagnostic and showcase plots

    Centerline velocity profiles for validation

## Notes

This implementation is intended for educational and reference purposes.
It emphasizes numerical clarity, physical correctness, and reproducibility.
It does not claim methodological novelty.
