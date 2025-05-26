"""Visualization module for annealing simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class Visualizer:
    """Class for visualizing annealing simulation results."""
    
    def __init__(self, grid_size):
        """Initialize visualizer."""
        self.grid_size = grid_size
        self.fig = None
        self.axes = None
        
        # Pre-allocate arrays for better performance
        self.empty_grid = np.zeros((grid_size, grid_size))
        self.empty_line = np.array([])
        
        # Initialize plot objects
        self.grain_image = None
        self.stress_image = None
        self.energy_line = None
        self.temperature_line = None
        self.grain_size_line = None
    
    def setup_plots(self):
        """Set up the visualization plots."""
        plt.style.use('bmh')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Annealing Simulation Progress', fontsize=16)
        
        # Initialize image objects
        self.grain_image = self.axes[0, 0].imshow(
            self.empty_grid, cmap='viridis', animated=True
        )
        self.axes[0, 0].set_title('Grain Structure')
        
        self.stress_image = self.axes[0, 1].imshow(
            self.empty_grid, cmap='coolwarm', animated=True
        )
        self.axes[0, 1].set_title('Residual Stress')
        
        # Initialize line plots
        self.energy_line, = self.axes[1, 0].plot([], [], 'b-', label='Energy')
        self.axes[1, 0].set_title('System Energy')
        self.axes[1, 0].legend()
        
        self.temperature_line, = self.axes[1, 1].plot([], [], 'r-', label='Temperature')
        self.axes[1, 1].set_title('Temperature')
        self.axes[1, 1].legend()
        
        # Adjust layout
        plt.tight_layout()
    
    def update_plots(self, frame_data):
        """Update plots with new simulation data."""
        try:
            # Update grain structure
            if frame_data['grid'] is not None:
                self.grain_image.set_data(frame_data['grid'])
            
            # Update stress field
            if frame_data['stress_field'] is not None:
                self.stress_image.set_data(frame_data['stress_field'])
            
            # Update line plots
            if frame_data['energy_history'] is not None:
                x = np.arange(len(frame_data['energy_history']))
                self.energy_line.set_data(x, frame_data['energy_history'])
                self.axes[1, 0].relim()
                self.axes[1, 0].autoscale_view()
            
            if frame_data['temperature_history'] is not None:
                x = np.arange(len(frame_data['temperature_history']))
                self.temperature_line.set_data(x, frame_data['temperature_history'])
                self.axes[1, 1].relim()
                self.axes[1, 1].autoscale_view()
            
            return (self.grain_image, self.stress_image, self.energy_line,
                    self.temperature_line)
        except Exception as e:
            print(f"Error updating plots: {e}")
            return None
    
    def create_animation(self, simulation_data, frames, interval=50):
        """Create animation of simulation progress."""
        self.setup_plots()
        
        def update(frame):
            try:
                if frame >= len(simulation_data):
                    return None
                return self.update_plots(simulation_data[frame])
            except Exception as e:
                print(f"Error in frame {frame}: {e}")
                return None
        
        # Create progress bar
        pbar = tqdm(total=frames, desc="Creating Animation")
        
        def update_progress(frame):
            pbar.update(1)
            return update(frame)
        
        anim = FuncAnimation(
            self.fig, update_progress,
            frames=min(frames, len(simulation_data)),
            interval=interval,
            blit=True
        )
        
        # Show the animation window
        plt.show(block=False)
        
        return anim
    
    def plot_final_results(self, simulation_data):
        """Plot final simulation results."""
        plt.figure(figsize=(12, 8))
        
        # Plot final grain structure
        plt.subplot(2, 2, 1)
        plt.imshow(simulation_data['grid'], cmap='viridis')
        plt.title('Final Grain Structure')
        plt.colorbar()
        
        # Plot final stress field
        plt.subplot(2, 2, 2)
        plt.imshow(simulation_data['stress_field'], cmap='coolwarm')
        plt.title('Final Residual Stress')
        plt.colorbar()
        
        # Plot grain size distribution
        plt.subplot(2, 2, 3)
        plt.hist(simulation_data['grain_sizes'], bins=20)
        plt.title('Grain Size Distribution')
        plt.xlabel('Grain Size')
        plt.ylabel('Count')
        
        # Plot energy evolution
        plt.subplot(2, 2, 4)
        plt.plot(simulation_data['energy_history'])
        plt.title('Energy Evolution')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        
        plt.tight_layout()
        plt.show(block=True)  # Block until window is closed 