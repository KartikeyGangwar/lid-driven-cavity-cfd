# Lid-Driven Cavity Flow Solver (2D)

High-resolution finite-difference solver for the 2D incompressible
lid-driven cavity problem using the streamfunction–vorticity formulation.

![Showcase](figures/lid_driven_showcase_Re1000_N251.png)

## Features
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport
- Red–Black SOR for Poisson equation
- Physically consistent coordinate system
- Instant post-processing from saved solutions (.npz)

## Usage

Run a simulation:
```bash
python lid_driven_cavity_fdm.py
```
Generate plots from a saved solution:
```bash
python plot_from_npz.py
```
Notes

This implementation is intended for educational and reference purposes.
It does not claim methodological novelty.
