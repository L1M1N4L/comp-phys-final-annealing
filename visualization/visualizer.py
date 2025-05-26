"""Visualization module for annealing simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

class Visualizer:
    """Class for visualizing annealing simulation results."""
    
    def __init__(self, grid_size):
        """Initialize visualizer.
        
        Args:
            grid_size (int): Size of the simulation grid
        """
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
        self.stress_line = None
    
    def setup_plots(self):
        """Set up the visualization plots."""
        plt.style.use('bmh')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
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
        self.energy_line, = self.axes[0, 2].plot([], [], 'b-', label='Energy')
        self.axes[0, 2].set_title('System Energy')
        self.axes[0, 2].legend()
        
        self.temperature_line, = self.axes[1, 0].plot([], [], 'r-', label='Temperature')
        self.axes[1, 0].set_title('Temperature')
        self.axes[1, 0].legend()
        
        self.grain_size_line, = self.axes[1, 1].plot([], [], 'g-', label='Grain Size')
        self.axes[1, 1].set_title('Average Grain Size')
        self.axes[1, 1].legend()
        
        self.stress_line, = self.axes[1, 2].plot([], [], 'm-', label='Residual Stress')
        self.axes[1, 2].set_title('Average Residual Stress')
        self.axes[1, 2].legend()
        
        # Adjust layout
        plt.tight_layout()
    
    def update_plots(self, frame_data):
        """Update plots with new simulation data.
        
        Args:
            frame_data (dict): Dictionary containing simulation data
        """
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
            self.axes[0, 2].relim()
            self.axes[0, 2].autoscale_view()
        
        if frame_data['temperature_history'] is not None:
            x = np.arange(len(frame_data['temperature_history']))
            self.temperature_line.set_data(x, frame_data['temperature_history'])
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
        
        if frame_data['grain_size_history'] is not None:
            x = np.arange(len(frame_data['grain_size_history']))
            self.grain_size_line.set_data(x, frame_data['grain_size_history'])
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()
        
        if frame_data['residual_stress_history'] is not None:
            x = np.arange(len(frame_data['residual_stress_history']))
            self.stress_line.set_data(x, frame_data['residual_stress_history'])
            self.axes[1, 2].relim()
            self.axes[1, 2].autoscale_view()
        
        return (self.grain_image, self.stress_image, self.energy_line,
                self.temperature_line, self.grain_size_line, self.stress_line)
    
    def create_animation(self, simulation_data, frames, interval=50):
        """Create animation of simulation progress.
        
        Args:
            simulation_data (list): List of simulation data dictionaries
            frames (int): Number of frames in animation
            interval (int): Interval between frames in milliseconds
            
        Returns:
            FuncAnimation: Animation object
        """
        self.setup_plots()
        
        def update(frame):
            return self.update_plots(simulation_data[frame])
        
        anim = FuncAnimation(
            self.fig, update,
            frames=tqdm(range(frames), desc="Creating Animation"),
            interval=interval,
            blit=True
        )
        
        # Show the animation window
        plt.show(block=False)
        
        return anim
    
    def plot_final_results(self, simulation_data):
        """Plot final simulation results.
        
        Args:
            simulation_data (dict): Dictionary containing final simulation data
        """
        plt.figure(figsize=(15, 10))
        
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