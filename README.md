# Annealing Simulation

A Python-based simulation of the annealing process for polycrystalline materials, incorporating advanced features such as rate sensitivity, component-specific soaking, cooling effects, abnormal grain detection, and residual stress prediction.

## Features

- Monte Carlo simulation of grain growth during annealing
- Material-specific properties and temperature profiles
- Component-specific soaking time calculations
- Residual stress prediction and visualization
- Abnormal grain detection and analysis
- Grain size distribution tracking and statistical analysis
- Comprehensive visualization of the annealing process

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/annealing.git
cd annealing
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python src/main.py
```

This will:
1. Initialize the simulation with default parameters
2. Run the annealing process
3. Generate visualizations in the `frames` directory
4. Save results in the `results` directory
5. Display inspection results in the console

## Project Structure

```
annealing/
├── src/
│   ├── models/
│   │   ├── annealing_simulation.py
│   │   ├── material_properties.py
│   │   ├── monte_carlo.py
│   │   └── temperature_controller.py
│   ├── analysis/
│   │   └── grain_analyzer.py
│   ├── visualization/
│   │   └── visualizer.py
│   ├── utils/
│   │   └── helpers.py
│   └── main.py
├── frames/
├── results/
├── requirements.txt
└── README.md
```

## Mathematical Formulation

### Basic Potts Model
The total system energy is given by:

$$E = \sum_{i,j} J_{ij}(1 - \delta_{s_i,s_j})$$

where:
- $J_{ij}$ is the interaction energy between sites $i$ and $j$
- $\delta_{s_i,s_j}$ is the Kronecker delta
- $s_i$ is the orientation at site $i$

### Extended Energy Model
The energy change for a proposed transition is:

$$\Delta E = E_{new} - E_{old} + \sigma_{residual}$$

where $\sigma_{residual}$ is the contribution from residual stress.

### Metropolis Algorithm
The transition probability is:

$$P = \begin{cases}
1 & \text{if } \Delta E \leq 0 \\
\exp(-\Delta E/k_BT) & \text{if } \Delta E > 0
\end{cases}$$

### Mobility-Weighted Metropolis Algorithm
The transition probability is modified by mobility:

$$P = \begin{cases}
M & \text{if } \Delta E \leq 0 \\
M\exp(-\Delta E/k_BT) & \text{if } \Delta E > 0
\end{cases}$$

where $M$ is the mobility factor.

### Grain Growth Kinetics
The average grain size evolution follows:

$$\bar{D}^n - \bar{D}_0^n = kt$$

where:
- $\bar{D}$ is the average grain size
- $n$ is the growth exponent
- $k$ is the rate constant
- $t$ is time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Potts model for grain growth
- Inspired by materials science research in annealing processes
- References:
  1. "Computer Simulation of Grain Growth" by Rollett et al.
  2. "Phase Field Methods in Materials Science and Engineering" by Chen
  3. "Grain Growth in Metals" by Humphreys and Hatherly
  4. "Materials Science and Engineering: An Introduction" by Callister
  5. "Physical Metallurgy Principles" by Reed-Hill and Abbaschian 