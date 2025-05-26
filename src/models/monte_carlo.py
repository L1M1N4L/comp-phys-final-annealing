"""Monte Carlo simulation module for annealing process."""

import numpy as np
from tqdm import tqdm

class MonteCarloSimulator:
    """Class for performing Monte Carlo simulation of annealing process."""
    
    def __init__(self, grid_size, material_properties, temperature_controller):
        """Initialize Monte Carlo simulator."""
        self.grid_size = grid_size
        self.material_properties = material_properties
        self.temperature_controller = temperature_controller
        self.boltzmann_constant = 8.617333262e-5  # eV/K
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Initialize history tracking
        self.energy_history = []
        self.temperature_history = []
        self.grain_size_history = []
    
    def create_initial_structure(self, num_grains):
        """Create initial polycrystalline structure with more diversity for visibility."""
        # Place random grain seeds
        positions = np.random.permutation(self.grid_size * self.grid_size)[:num_grains]
        x = positions % self.grid_size
        y = positions // self.grid_size
        self.grid[:, :] = np.random.randint(1, num_grains + 1, size=(self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[y, x] = np.arange(1, num_grains + 1, dtype=np.int32)
    
    def calculate_energy(self, i, j):
        """Calculate local energy at a site."""
        current = self.grid[i, j]
        energy = 0.0
        
        # Check 4 nearest neighbors
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = (i + di) % self.grid_size, (j + dj) % self.grid_size
            if self.grid[ni, nj] != current:
                energy += 1.0
        
        # INCREASE boundary energy for visibility
        return energy * max(self.material_properties.boundary_energy, 1.0)
    
    def monte_carlo_step(self, grain_analyzer):
        """Perform one Monte Carlo step with more visible grain evolution."""
        self.temperature = self.temperature_controller.update_temperature()
        attempts = self.grid_size * self.grid_size
        for _ in range(attempts):
            i = np.random.randint(0, self.grid_size)
            j = np.random.randint(0, self.grid_size)
            current = self.grid[i, j]
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = (i + di) % self.grid_size, (j + dj) % self.grid_size
                neighbors.append(self.grid[ni, nj])
            if not neighbors:
                continue
            new_orientation = neighbors[np.random.randint(0, len(neighbors))]
            old_energy = self.calculate_energy(i, j)
            self.grid[i, j] = new_orientation
            new_energy = self.calculate_energy(i, j)
            delta_energy = new_energy - old_energy
            # Metropolis criterion: allow changes if delta_energy <= 0, or with probability if >0
            if delta_energy > 0:
                kT = self.boltzmann_constant * self.temperature
                if kT == 0 or np.random.random() > np.exp(-delta_energy / kT):
                    self.grid[i, j] = current
        # Print unique values for diagnosis
        print(f"Unique grains after step: {np.unique(self.grid)}")
        self._update_statistics(grain_analyzer)
    
    def _update_statistics(self, grain_analyzer):
        """Update simulation statistics."""
        # Calculate total energy
        total_energy = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                total_energy += self.calculate_energy(i, j)
        
        # Update history
        self.energy_history.append(total_energy)
        self.temperature_history.append(self.temperature)
        
        # Calculate grain sizes (every 10 steps)
        if len(self.grain_size_history) % 10 == 0:
            grain_sizes = grain_analyzer.calculate_grain_sizes(self.grid)
            avg_grain_size = np.mean(grain_sizes) if grain_sizes.size > 0 else 0.0
            self.grain_size_history.append(avg_grain_size) 