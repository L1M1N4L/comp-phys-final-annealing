import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import time
import sys
import os

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
    
    def run_simulation(self, save_frames=True, frame_dir='frames'):
        """Run the complete annealing simulation."""
        print("Starting annealing simulation...")
        start_time = time.time()
        frames = []
        
        # Create directory for frames if saving
        if save_frames:
            os.makedirs(frame_dir, exist_ok=True)
        
        # Create a custom colormap with distinct colors for grains
        num_colors = self.num_grains + 1
        colors = plt.cm.jet(np.linspace(0, 1, num_colors))
        np.random.shuffle(colors)  # Shuffle colors for better visualization
        cmap = ListedColormap(colors)
        
        # Save initial frame
        frames.append(self.grid.copy())
        
        # Save every frame_interval iterations
        frame_interval = max(1, self.max_iterations // 50)  # Save ~50 frames
        
        # Run simulation for max_iterations
        for i in range(self.max_iterations):
            self.monte_carlo_step()

            # Save frame at intervals
            if i % frame_interval == 0 or i == self.max_iterations - 1:
                frames.append(self.grid.copy())
                
                # Save individual frame as image if requested
                if save_frames:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(self.grid, cmap=cmap)
                    temp_text = f"Temperature: {self.current_temperature:.1f}K"
                    
                    # Determine phase
                    phase = "Heating"
                    if i >= self.max_iterations // 4 and i < self.max_iterations // 2:
                        phase = "Soaking"
                    elif i >= self.max_iterations // 2:
                        phase = "Cooling"
                    
                    plt.title(f'Annealing Simulation\nIteration {i+1}/{self.max_iterations} - {phase}\n{temp_text}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{frame_dir}/frame_{i:04d}.png', dpi=150)
                    plt.close()
                
            # Print progress
            if i % (self.max_iterations // 20) == 0 or i == self.max_iterations - 1:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * self.max_iterations
                remaining = estimated_total - elapsed
                
                sys.stdout.write(f"\rProgress: {i+1}/{self.max_iterations} ({(i+1)/self.max_iterations*100:.1f}%) " + 
                                 f"- Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
                sys.stdout.flush()
        
        print("\nSimulation completed in {:.2f} seconds!".format(time.time() - start_time))
        
        # Ensure we have complete statistics
        if len(self.energy_history) < self.max_iterations:
            # Fill in missing values by linear interpolation
            self.energy_history = np.interp(
                np.linspace(0, 1, self.max_iterations),
                np.linspace(0, 1, len(self.energy_history)), 
                self.energy_history
            ).tolist()
            
            self.average_grain_size_history = np.interp(
                np.linspace(0, 1, self.max_iterations),
                np.linspace(0, 1, len(self.average_grain_size_history)), 
                self.average_grain_size_history
            ).tolist()
            
            self.temperature_history = np.interp(
                np.linspace(0, 1, self.max_iterations),
                np.linspace(0, 1, len(self.temperature_history)), 
                self.temperature_history
            ).tolist()
        
        return frames
    
    def create_animation_gif(self, frames, filename='annealing_animation.gif', fps=10):
        """Create animation from simulation frames as a GIF (no FFmpeg required)."""
        print("Creating animation GIF...")
        
        # Create a custom colormap with distinct colors for grains
        num_colors = self.num_grains + 1
        colors = plt.cm.jet(np.linspace(0, 1, num_colors))
        np.random.shuffle(colors)  # Shuffle colors for better visualization
        cmap = ListedColormap(colors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # List to store image objects for animation
        images = []
        
        # Create frames for animation
        for i, frame in enumerate(frames):
            ax.clear()
            img = ax.imshow(frame, cmap=cmap)
            
            # Determine phase based on frame index
            phase = "Heating"
            if i >= len(frames) // 4 and i < len(frames) // 2:
                phase = "Soaking"
            elif i >= len(frames) // 2:
                phase = "Cooling"
            
            # Map frame index to temperature profile index
            temp_idx = min(self.max_iterations - 1, int(i * self.max_iterations / len(frames)))
            temp_text = f"Temperature: {self.temperature_profile[temp_idx]:.1f}K"
            
            ax.set_title(f'Annealing Simulation\nFrame {i+1}/{len(frames)} - {phase}\n{temp_text}')
            ax.axis('off')
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            
            # Print progress
            if i % (len(frames) // 10) == 0 or i == len(frames) - 1:
                sys.stdout.write(f"\rProcessing frames: {i+1}/{len(frames)} ({(i+1)/len(frames)*100:.1f}%)")
                sys.stdout.flush()
        
        print("\nSaving animation as GIF...")
        
        # Create animation from images
        try:
            from PIL import Image
            
            # Convert images to PIL Images and save as GIF
            pil_images = [Image.fromarray(img) for img in images]
            
            # Save as GIF
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                optimize=False,
                duration=int(1000/fps),  # milliseconds per frame
                loop=0  # Loop forever
            )
            
            print(f"Animation saved as '{filename}'")
            
        except ImportError:
            print("PIL (Pillow) library not found. Saving individual frames only.")
            print("To create an animation, install Pillow: pip install Pillow")
    
    def create_animated_plot(self, frames, filename='annealing_with_plots.gif', fps=10):
        """Create an animation with the microstructure and plots side by side."""
        print("Creating animated visualization with plots...")
        
        # Ensure we have enough data points
        if len(self.energy_history) < self.max_iterations:
            # Fill in missing values by linear interpolation
            self.energy_history = np.interp(
                np.linspace(0, 1, self.max_iterations),
                np.linspace(0, 1, len(self.energy_history)), 
                self.energy_history
            ).tolist()
            
            self.temperature_history = np.interp(
                np.linspace(0, 1, self.max_iterations),
                np.linspace(0, 1, len(self.temperature_history)), 
                self.temperature_history
            ).tolist()
        
        # Subsample the history data to match the number of frames
        subsample_indices = np.linspace(0, len(self.temperature_history)-1, len(frames)).astype(int)
        subsampled_temp = [self.temperature_history[i] for i in subsample_indices]
        subsampled_energy = [self.energy_history[i] for i in subsample_indices]
        
        # Create a custom colormap with distinct colors for grains
        num_colors = self.num_grains + 1
        colors = plt.cm.jet(np.linspace(0, 1, num_colors))
        np.random.shuffle(colors)  # Shuffle colors for better visualization
        cmap = ListedColormap(colors)
        
        try:
            from PIL import Image
            
            # Create figure
            fig = plt.figure(figsize=(16, 8))
            
            # List to store image objects for animation
            images = []
            
            # Create frames for animation
            for i, frame in enumerate(frames):
                plt.clf()  # Clear figure
                
                # Left subplot - microstructure
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(frame, cmap=cmap)
                
                # Determine phase based on frame index
                phase = "Heating"
                if i >= len(frames) // 4 and i < len(frames) // 2:
                    phase = "Soaking"
                elif i >= len(frames) // 2:
                    phase = "Cooling"
                
                ax1.set_title(f'Microstructure\nFrame {i+1}/{len(frames)} - {phase}\nTemperature: {subsampled_temp[i]:.1f}K')
                ax1.axis('off')
                
                # Right subplot - plots
                ax2 = plt.subplot(1, 2, 2)
                
                # Plot temperature
                ax2.plot(subsampled_temp[:i+1], 'r-', label='Temperature (K)')
                
                # Plot energy on secondary y-axis
                ax3 = ax2.twinx()
                ax3.plot(subsampled_energy[:i+1], 'g-', label='Energy')
                
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Temperature (K)', color='r')
                ax3.set_ylabel('Energy', color='g')
                
                # Add vertical line at current frame
                ax2.axvline(x=i, color='b', linestyle='--')
                
                # Create custom legend
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax3.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                ax2.grid(True)
                
                plt.tight_layout()
                
                # Convert plot to image
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
                
                # Print progress
                if i % (len(frames) // 10) == 0 or i == len(frames) - 1:
                    sys.stdout.write(f"\rProcessing frames: {i+1}/{len(frames)} ({(i+1)/len(frames)*100:.1f}%)")
                    sys.stdout.flush()
            
            print("\nSaving animation as GIF...")
            
            # Convert images to PIL Images and save as GIF
            pil_images = [Image.fromarray(img) for img in images]
            
            # Save as GIF
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                optimize=False,
                duration=int(1000/fps),  # milliseconds per frame
                loop=0  # Loop forever
            )
            
            print(f"Animation with plots saved as '{filename}'")
            
        except ImportError:
            print("PIL (Pillow) library not found. Cannot create animated plot.")
            print("To create an animation, install Pillow: pip install Pillow")
    
    def visualize_results(self, frames):
        """Visualize simulation results with before/after microstructure and physical parameters."""
        # Create a custom colormap with distinct colors for grains
        num_colors = max(np.max(self.initial_grid), np.max(self.grid)) + 1
        colors = plt.cm.jet(np.linspace(0, 1, num_colors))
        np.random.shuffle(colors)  # Shuffle colors for better visualization
        cmap = ListedColormap(colors)
        
        # Create figure for visualizations
        plt.figure(figsize=(16, 12))
        
        # Initial microstructure
        plt.subplot(2, 3, 1)
        plt.imshow(self.initial_grid, cmap=cmap)
        plt.title('Initial Microstructure')
        plt.axis('off')
        
        # Final microstructure
        plt.subplot(2, 3, 2)
        plt.imshow(self.grid, cmap=cmap)
        plt.title('Final Microstructure')
        plt.axis('off')
        
        # Temperature profile
        plt.subplot(2, 3, 3)
        plt.plot(self.temperature_history, 'r-')
        plt.title('Temperature Profile')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature (K)')
        plt.grid(True)
        
        # System energy
        plt.subplot(2, 3, 4)
        plt.plot(self.energy_history, 'g-')
        plt.title('System Energy Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Total Energy (a.u.)')
        plt.grid(True)
        
        # Average grain size
        plt.subplot(2, 3, 5)
        plt.plot(self.average_grain_size_history, 'b-')
        plt.title('Average Grain Size Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Average Grain Size (pixels)')
        plt.grid(True)
        
        # Combined plot
        plt.subplot(2, 3, 6)
        plt.plot(np.array(self.energy_history) / max(self.energy_history), 'g-', label='Normalized Energy')
        plt.plot(np.array(self.temperature_history) / max(self.temperature_history), 'r-', label='Normalized Temperature')
        plt.title('Energy vs Temperature')
        plt.xlabel('Iteration')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('annealing_simulation_results.png', dpi=300)
        plt.show()
        
        # Create snapshots of the evolution
        plt.figure(figsize=(15, 3))
        num_frames = len(frames)
        for i, frame_idx in enumerate(range(0, num_frames, max(1, num_frames // 5))):
            if i >= 5:  # Show at most 5 frames
                break
            if frame_idx < num_frames:
                plt.subplot(1, 5, i+1)
                plt.imshow(frames[frame_idx], cmap=cmap)
                phase = "Heating" if frame_idx < num_frames//3 else "Soaking" if frame_idx < 2*num_frames//3 else "Cooling"
                plt.title(f'{phase}\nStep {frame_idx}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('annealing_evolution.png', dpi=300)
        plt.show()

# Main function to run the simulation
def main():
    # Use smaller values for a faster simulation
    sim = AnnealingSimulation(
        grid_size=50,     # Size of the grid
        num_grains=20,    # Number of initial grains
        max_iterations=200  # Number of simulation steps
    )
    
    # Run simulation and save frames
    frames = sim.run_simulation(save_frames=True, frame_dir='annealing_frames')
    
    # Create animations (using multiple methods to ensure at least one works)
    
    # Method 1: Create simple GIF animation (no FFmpeg needed)
    sim.create_animation_gif(frames, filename='annealing_animation.gif', fps=10)
    
    # Method 2: Create more advanced animation with plots (if PIL is available)
    sim.create_animated_plot(frames, filename='annealing_with_plots.gif', fps=8)
    
    # Visualize results
    sim.visualize_results(frames)
    
    print("Simulation, animation, and visualization complete!")
    print("Animations saved as GIF files")
    print("Individual frames saved in 'annealing_frames' directory")

if __name__ == "__main__":
    main()