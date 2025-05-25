import numpy as np
import sys
import time

class AnnealingSimulation:
    def __init__(self, grid_size=50, num_grains=20, max_iterations=500):
        """Initialize the annealing simulation with a polycrystalline microstructure."""
        self.grid_size = grid_size
        self.num_grains = num_grains
        self.max_iterations = max_iterations
        
        # Create initial microstructure with random grains
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.create_initial_structure()
        
        # Physical parameters
        self.temperature_profile = self.create_temperature_profile()
        self.current_temp_idx = 0
        self.current_temperature = self.temperature_profile[0]
        self.boltzmann_constant = 1.0  # Normalized for simulation
        self.boundary_energy = 1.0     # Energy at grain boundaries
        
        # Tracking statistics
        self.energy_history = []
        self.average_grain_size_history = []
        self.temperature_history = []
        
        # Save initial state
        self.initial_grid = self.grid.copy()
    
    def create_initial_structure(self):
        """Create initial polycrystalline structure with random grain orientations."""
        # Place random grain seeds
        for i in range(1, self.num_grains + 1):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            self.grid[y, x] = i
        
        # Grow grains outward (simple cellular automaton approach)
        iterations = 0
        max_fill_iterations = self.grid_size * 2  # Limit iterations to prevent infinite loops
        
        while 0 in self.grid and iterations < max_fill_iterations:
            iterations += 1
            new_grid = self.grid.copy()
            
            # Find empty cells with neighbors
            empty_cells = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.grid[y, x] == 0:  # Empty cell
                        has_neighbor = False
                        neighbor_values = []
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] != 0:
                                has_neighbor = True
                                neighbor_values.append(self.grid[ny, nx])
                        if has_neighbor:
                            empty_cells.append((x, y, neighbor_values))
            
            # Fill empty cells with neighbors
            for x, y, neighbors in empty_cells:
                new_grid[y, x] = np.random.choice(neighbors)
            
            self.grid = new_grid
            
            # Progress update for initialization
            if iterations % 5 == 0:
                filled = np.count_nonzero(self.grid)
                total = self.grid_size * self.grid_size
                percent_filled = filled / total * 100
                sys.stdout.write(f"\rInitializing microstructure: {percent_filled:.1f}% filled")
                sys.stdout.flush()
        
        # Fill any remaining empty cells with random grain IDs
        if 0 in self.grid:
            empty_mask = (self.grid == 0)
            random_grains = np.random.randint(1, self.num_grains + 1, size=np.count_nonzero(empty_mask))
            self.grid[empty_mask] = random_grains
        
        print("\nInitialization complete!")
    
    def create_temperature_profile(self):
        """Create temperature profile for annealing process: heating, soaking, cooling."""
        # Create a realistic annealing temperature profile
        ramp_up = np.linspace(300, 1000, self.max_iterations // 4)  # Heating phase (K)
        soak = np.ones(self.max_iterations // 4) * 1000  # Soaking phase (K)
        cool_slow = np.linspace(1000, 600, self.max_iterations // 4)  # Slow cooling phase (K)
        cool_fast = np.linspace(600, 300, self.max_iterations // 4)  # Fast cooling phase (K)
        
        return np.concatenate((ramp_up, soak, cool_slow, cool_fast))
    
    def monte_carlo_step(self):
        """Perform one Monte Carlo step of the simulation."""
        # Update temperature according to profile
        if self.current_temp_idx < len(self.temperature_profile) - 1:
            self.current_temp_idx += 1
            self.current_temperature = self.temperature_profile[self.current_temp_idx]
        
        # Sample a subset of grid points for faster simulation
        num_samples = self.grid_size * self.grid_size // 10  # Sample 10% of the grid
        
        for _ in range(num_samples):
            # Select random site
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            
            # Get neighboring grain orientations
            neighbor_grains = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbor_grains.append(self.grid[ny, nx])
            
            if not neighbor_grains:
                continue
            
            # Choose random orientation from neighbors
            current_orientation = self.grid[y, x]
            new_orientation = np.random.choice(neighbor_grains)
            
            if new_orientation != current_orientation:
                # Calculate energy change (simplified)
                energy_before = 0
                energy_after = 0
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        neighbor = self.grid[ny, nx]
                        if neighbor != current_orientation:
                            energy_before += 1
                        if neighbor != new_orientation:
                            energy_after += 1
                
                # Calculate energy difference
                delta_E = energy_after - energy_before
                
                # Apply Metropolis algorithm
                if delta_E <= 0 or np.random.random() < np.exp(-delta_E / (self.boltzmann_constant * self.current_temperature)):
                    # Accept the new orientation
                    self.grid[y, x] = new_orientation
        
        # Calculate system properties for tracking (do this less frequently)
        if self.current_temp_idx % 5 == 0:
            total_energy = self.calculate_total_energy()
            avg_grain_size = self.calculate_average_grain_size()
            
            self.energy_history.append(total_energy)
            self.average_grain_size_history.append(avg_grain_size)
            self.temperature_history.append(self.current_temperature)
    
    def calculate_total_energy(self):
        """Calculate total energy of the system (simplified)."""
        total_energy = 0
        
        # Sample points to estimate energy
        sample_points = 100
        for _ in range(sample_points):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            center_grain = self.grid[y, x]
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[ny, nx] != center_grain:
                        total_energy += 1
        
        return total_energy / sample_points  # Normalized energy estimate
    
    def calculate_average_grain_size(self):
        """Estimate average grain size by sampling."""
        # Sample a subset of points and flood fill to estimate sizes
        sample_size = 10
        grain_sizes = []
        
        for _ in range(sample_size):
            x, y = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
            grain_id = self.grid[y, x]
            
            # Count contiguous cells with same grain ID (simplified)
            count = 0
            for i in range(max(0, x-5), min(self.grid_size, x+6)):
                for j in range(max(0, y-5), min(self.grid_size, y+6)):
                    if self.grid[j, i] == grain_id:
                        count += 1
            
            grain_sizes.append(count)
        
        return np.mean(grain_sizes) if grain_sizes else 0
    
    def run_simulation(self):
        """Run the complete annealing simulation."""
        print("Starting annealing simulation...")
        start_time = time.time()
        frames = []
        
        for i in range(self.max_iterations):
            self.monte_carlo_step()
            
            # Save frame every few iterations
            if i % 5 == 0:
                frames.append(self.grid.copy())
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / self.max_iterations
                remaining = elapsed / progress - elapsed
                sys.stdout.write(f"\rProgress: {i+1}/{self.max_iterations} ({progress*100:.1f}%) - "
                               f"Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
                sys.stdout.flush()
        
        print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds!")
        return frames 