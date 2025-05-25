# Monte Carlo Approach to Metal Annealing

This project implements a Monte Carlo simulation of the annealing process in polycrystalline materials, providing insights into grain growth, recrystallization, and microstructure evolution.

## Overview

### Definition and Purpose
Annealing is a controlled heat treatment process where metals are heated to a specific temperature and then cooled at a controlled rate. This simulation helps understand and predict the microstructural changes during this process.

### Physical Process Breakdown
1. **Heating Phase**
   - Material is heated above recrystallization temperature
   - Enables atomic diffusion and dislocation recovery
   - Typical heating rates:
     * Aluminum alloys: 3-5°C/min
     * Steels: 10-15°C/min
     * Titanium alloys: 5-8°C/min

2. **Soaking Phase**
   - Material is held at high temperature
   - Allows grain nucleation and growth
   - Enables atomic diffusion
   - Duration depends on:
     * Material thickness
     * Desired grain size
     * Alloy composition

3. **Cooling Phase**
   - Controlled cooling determines final microstructure
   - Slow cooling → equilibrium structures
   - Faster cooling → enhanced properties (hardness, toughness)
   - Cooling rates affect:
     * Grain size distribution
     * Phase transformations
     * Residual stresses

## Theoretical Background

### Crystal Structure and Defects
1. **Crystal Lattice**
   - Face-centered cubic (FCC)
   - Body-centered cubic (BCC)
   - Hexagonal close-packed (HCP)

2. **Defects**
   - Point defects (vacancies, interstitials)
   - Line defects (dislocations)
   - Planar defects (grain boundaries)
   - Volume defects (voids, inclusions)

### Grain Boundary Physics
1. **Types of Boundaries**
   - Low-angle boundaries (< 15°)
   - High-angle boundaries (> 15°)
   - Special boundaries (Σ3, Σ5, etc.)

2. **Boundary Energy**
   - Orientation-dependent energy
   - Temperature dependence
   - Impurity effects

## Mathematical Formulation

### 1. Total System Energy (Basic Potts Model)
The system is represented as a lattice of sites, where each site has an orientation (spin) σ. The total system energy E is calculated as:

$$
E = \sum_{\langle i,j \rangle} J(1 - \delta_{\sigma_i, \sigma_j})
$$

Where:
- $\langle i, j \rangle$: Neighboring lattice site pairs
- $\sigma_i$: Orientation (spin) at site i
- $J$: Interaction energy constant
- $\delta_{\sigma_i, \sigma_j}$: Kronecker delta (1 if $\sigma_i = \sigma_j$, 0 otherwise)

### 2. Extended Energy Model
For more accurate physical behavior:

$$
E = \frac{1}{2} \sum_{j=1}^{N} \sum_{i=1}^{n} \{ \gamma(S_i, S_j)(1 - \delta_{S_i, S_j}) + F(S_j) \}
$$


Where:
- $N$: Total number of lattice sites
- $n$: Number of neighbors per site
- $S_i, S_j$: Grain orientations
- $\gamma(S_i, S_j)$: Orientation-dependent grain boundary energy
- $F(S_j)$: Stored energy at site j

### 3. Transition Probability (Metropolis Algorithm)
The Metropolis acceptance rule:

$$
P(\text{accept}) = \begin{cases} 
1, & \Delta E \leq 0 \\ 
\exp\left(-\frac{\Delta E}{kT}\right), & \Delta E > 0 
\end{cases}
$$

Where:
- $\Delta E$: Energy change
- $k$: Boltzmann constant
- $T$: Simulation temperature

### 4. Mobility-Weighted Metropolis Algorithm
Extended transition rule incorporating grain boundary mobility:

$$
P = \begin{cases} 
\frac{\gamma(S_i, S_j)}{\gamma_{\max}} \cdot \frac{\mu(S_i, S_j)}{\mu_{\max}}, & \Delta E \leq 0 \\ 
\frac{\gamma(S_i, S_j)}{\gamma_{\max}} \cdot \frac{\mu(S_i, S_j)}{\mu_{\max}} \cdot \exp\left(-\frac{\Delta E}{T}\right), & \Delta E > 0 
\end{cases}
$$

### 5. Grain Growth Kinetics
The rate of grain growth follows the relationship:

$$
\frac{dR}{dt} = M \cdot \frac{\gamma}{R}
$$

Where:
- $R$: Grain radius
- $M$: Grain boundary mobility
- $\gamma$: Grain boundary energy
- $t$: Time

## Implementation Details

### 1. Grid Representation
- 2D square lattice
- Periodic boundary conditions
- Variable grid resolution
- Memory-efficient data structures

### 2. Monte Carlo Steps
1. **Site Selection**
   - Random site selection
   - Weighted selection based on energy
   - Boundary-focused sampling

2. **Energy Calculation**
   - Local energy computation
   - Efficient neighbor lookup
   - Cached energy values

3. **State Updates**
   - Atomic operations
   - Parallel processing
   - Checkpointing

### 3. Performance Optimizations
1. **Computational**
   - Multi-threading
   - SIMD operations
   - Cache-aware algorithms

2. **Memory**
   - Sparse matrix storage
   - Memory mapping
   - Incremental updates

## Validation and Verification

### 1. Analytical Validation
- Comparison with analytical solutions
- Energy conservation checks
- Boundary condition verification

### 2. Experimental Validation
- Comparison with experimental data
- Grain size distribution analysis
- Texture evolution verification

### 3. Convergence Studies
- Grid size dependence
- Time step sensitivity
- Statistical sampling

## Applications

### Industrial Applications
1. **Heat Treatment Parameter Design**
   - Optimal heating rates
   - Soaking time optimization
   - Cooling rate control
   - Process window determination

2. **Microstructure Tailoring**
   - Grain size control (5-12µm range)
   - Distribution management
   - Texture development
   - Phase transformation control

3. **Defect Prevention**
   - Abnormal grain growth prediction
   - Residual stress reduction
   - Quality control criteria
   - Process optimization

### Computational Aspects
1. **Parallel Processing**
   - Multi-threading optimization
   - Memory management
   - Cache-aware algorithms
   - Load balancing

2. **Storage and I/O**
   - Checkpointing
   - Incremental storage
   - Real-time visualization
   - Data compression

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation using:
```bash
python src/main.py
```

The program will:
1. Create an initial microstructure
2. Run the annealing simulation
3. Generate two animation files in the `frames` directory:
   - `annealing_animation.gif`: Basic animation of the annealing process
   - `annealing_with_plots.gif`: Detailed visualization with plots

## Parameters

You can modify the simulation parameters in `src/main.py`:
- `grid_size`: Size of the simulation grid (default: 50)
- `num_grains`: Number of initial grains (default: 20)
- `max_iterations`: Number of simulation steps (default: 200)

## Output

The simulation generates two types of visualizations:
1. A simple animation showing the evolution of the microstructure
2. A detailed visualization with plots showing:
   - Microstructure evolution
   - System energy over time
   - Average grain size and temperature profiles

## References

1. Monte Carlo Methods in Statistical Physics
2. Grain Growth in Polycrystalline Materials
3. Computer Simulation of Microstructural Evolution
4. Phase Transformations in Metals and Alloys
5. Materials Science and Engineering: An Introduction 