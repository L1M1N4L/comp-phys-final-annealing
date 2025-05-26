"""Monte Carlo simulation module for annealing process."""

import numpy as np
from tqdm import tqdm

class MonteCarloSimulator:
    """Class for performing Monte Carlo simulation of annealing process."""
    
    def __init__(self, grid_size, material_properties, temperature_controller):
        """Initialize Monte Carlo simulator.
        
        Args:
            grid_size (int): Size of the simulation grid
            material_properties (MaterialProperties): Material properties object
            temperature_controller (TemperatureController): Temperature controller object
        """
        self.grid_size = grid_size
        self.material_properties = material_properties
        self.temperature_controller = temperature_controller
        self.boltzmann_constant = 8.617333262e-5  # eV/K
        
        # Initialize grid and energy storage
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.stored_energy = np.zeros((grid_size, grid_size))
        
        # Initialize energy and temperature
        self.energy = 0.0
        self.temperature = self.temperature_controller.current_temperature
        
        # Initialize history tracking
        self.energy_history = []
        self.temperature_history = []
        self.residual_stress_history = []
        self.grain_size_history = []
        self.abnormal_grain_history = []
    
    def create_initial_structure(self, num_grains):
        """Create initial polycrystalline structure.
        
        Args:
            num_grains (int): Number of initial grains
        """
        # Place random grain seeds
        for i in range(1, num_grains + 1):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            self.grid[y, x] = i
        
        # Grow grains outward
        iterations = 0
        max_fill_iterations = self.grid_size * 2
        
        while 0 in self.grid and iterations < max_fill_iterations:
            iterations += 1
            new_grid = self.grid.copy()
            
            # Find empty cells with neighbors
            empty_cells = self._find_empty_cells_with_neighbors()
            
            # Fill empty cells with neighbors
            for x, y, neighbors in empty_cells:
                new_grid[y, x] = np.random.choice(neighbors)
            
            self.grid = new_grid
        
        # Fill any remaining empty cells
        if 0 in self.grid:
            empty_mask = (self.grid == 0)
            random_grains = np.random.randint(1, num_grains + 1, size=np.count_nonzero(empty_mask))
            self.grid[empty_mask] = random_grains
        
        # Calculate initial energy
        self.energy = self.calculate_energy()
    
    def calculate_energy(self):
        """Calculate the total energy of the system.
        
        Returns:
            float: Total system energy
        """
        energy = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get current grain orientation
                current = self.grid[i, j]
                
                # Check neighbors (periodic boundary conditions)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = (i + di) % self.grid_size
                    nj = (j + dj) % self.grid_size
                    neighbor = self.grid[ni, nj]
                    
                    # Add energy if orientations are different
                    if current != neighbor:
                        energy += self.material_properties.boundary_energy
        
        return energy
    
    def monte_carlo_step(self, grain_analyzer):
        """Perform one Monte Carlo step of the simulation.
        
        Args:
            grain_analyzer (GrainAnalyzer): Grain analyzer object
        """
        # Update temperature
        self.temperature = self.temperature_controller.update_temperature()
        
        # Calculate current residual stress
        current_stress, stress_field = grain_analyzer.calculate_residual_stress(
            self.grid, self.temperature, self.material_properties
        )
        self.residual_stress_history.append(current_stress)
        
        # Detect abnormal grains
        num_abnormal, _ = grain_analyzer.detect_abnormal_grains(self.grid)
        self.abnormal_grain_history.append(num_abnormal)
        
        # Perform multiple attempts per step
        for _ in range(self.grid_size * self.grid_size):
            # Randomly select a site
            i = np.random.randint(0, self.grid_size)
            j = np.random.randint(0, self.grid_size)
            
            # Get current orientation
            current = self.grid[i, j]
            
            # Get neighbor orientations
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni = (i + di) % self.grid_size
                nj = (j + dj) % self.grid_size
                neighbors.append(self.grid[ni, nj])
            
            # Calculate energy change
            old_energy = sum(1 for n in neighbors if n != current)
            new_orientation = np.random.choice(neighbors)
            new_energy = sum(1 for n in neighbors if n != new_orientation)
            delta_energy = new_energy - old_energy
            
            # Add stress contribution
            stress_contribution = stress_field[i, j] / self.material_properties.youngs_modulus
            delta_energy += stress_contribution
            
            # Accept or reject change based on Metropolis criterion
            if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / (self.boltzmann_constant * self.temperature)):
                self.grid[i, j] = new_orientation
                self.energy += delta_energy
        
        # Update statistics
        self._update_statistics(grain_analyzer)
    
    def _update_statistics(self, grain_analyzer):
        """Update simulation statistics.
        
        Args:
            grain_analyzer (GrainAnalyzer): Grain analyzer object
        """
        # Calculate grain sizes
        grain_sizes = grain_analyzer.calculate_grain_sizes(self.grid)
        avg_grain_size = np.mean(grain_sizes) if grain_sizes.size > 0 else 0.0
        
        # Update history
        self.energy_history.append(self.energy)
        self.temperature_history.append(self.temperature)
        self.grain_size_history.append(avg_grain_size)
    
    def _find_empty_cells_with_neighbors(self):
        """Find empty cells that have neighboring grains.
        
        Returns:
            list: List of (x, y, neighbors) tuples
        """
        empty_cells = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == 0:  # Empty cell
                    has_neighbor = False
                    neighbor_values = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                            self.grid[ny, nx] != 0):
                            has_neighbor = True
                            neighbor_values.append(self.grid[ny, nx])
                    if has_neighbor:
                        empty_cells.append((x, y, neighbor_values))
        return empty_cells 