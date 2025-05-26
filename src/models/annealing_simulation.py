"""Main annealing simulation module."""

import numpy as np
from tqdm import tqdm
from src.models.material_properties import MaterialProperties
from src.models.temperature_controller import TemperatureController
from src.models.monte_carlo import MonteCarloSimulator
from src.analysis.grain_analyzer import GrainAnalyzer

class AnnealingSimulation:
    """Main class for coordinating the annealing simulation."""
    
    def __init__(self, grid_size=100, num_grains=30, max_iterations=1000,
                 material_type='aluminum', component_thickness=10.0, component_area=100.0):
        """Initialize the annealing simulation.
        
        Args:
            grid_size (int): Size of the simulation grid
            num_grains (int): Number of initial grains
            max_iterations (int): Maximum number of Monte Carlo steps
            material_type (str): Type of material ('aluminum' or 'steel')
            component_thickness (float): Thickness of the component in mm
            component_area (float): Surface area of the component in mm²
        """
        # Initialize components
        self.material_properties = MaterialProperties(material_type)
        self.temperature_controller = TemperatureController(
            self.material_properties,
            max_iterations,
            component_thickness,
            component_area
        )
        self.monte_carlo = MonteCarloSimulator(
            grid_size,
            self.material_properties,
            self.temperature_controller
        )
        self.grain_analyzer = GrainAnalyzer(grid_size)
        
        # Simulation parameters
        self.grid_size = grid_size
        self.num_grains = num_grains
        self.max_iterations = max_iterations
        
        # Initialize simulation
        self.monte_carlo.create_initial_structure(num_grains)
        
        # Initialize history tracking
        self.grid_history = []
        self.statistics_history = []
    
    def run(self):
        """Run the complete annealing simulation.
        
        Returns:
            tuple: (final_grid, statistics)
        """
        # Initialize statistics
        statistics = {
            'energy_history': [],
            'temperature_history': [],
            'grain_size_history': [],
            'residual_stress_history': [],
            'abnormal_grain_history': [],
            'grid_history': [],
            'statistics_history': []
        }
        
        # Run simulation with progress bar
        for iteration in tqdm(range(self.max_iterations), desc="Annealing Progress"):
            # Perform Monte Carlo step
            self.monte_carlo.monte_carlo_step(self.grain_analyzer)
            
            # Calculate current statistics
            current_stats = {
                'energy': self.monte_carlo.energy,
                'temperature': self.monte_carlo.temperature,
                'grain_sizes': self.grain_analyzer.calculate_grain_sizes(self.monte_carlo.grid),
                'residual_stress': self.grain_analyzer.calculate_residual_stress(
                    self.monte_carlo.grid,
                    self.monte_carlo.temperature,
                    self.material_properties
                )[0],
                'abnormal_grains': self.grain_analyzer.detect_abnormal_grains(self.monte_carlo.grid)[0]
            }
            
            # Update history
            statistics['energy_history'].append(current_stats['energy'])
            statistics['temperature_history'].append(current_stats['temperature'])
            statistics['grain_size_history'].append(np.mean(current_stats['grain_sizes']))
            statistics['residual_stress_history'].append(current_stats['residual_stress'])
            statistics['abnormal_grain_history'].append(current_stats['abnormal_grains'])
            statistics['grid_history'].append(self.monte_carlo.grid.copy())
            statistics['statistics_history'].append(current_stats)
        
        # Collect final statistics
        final_stats = {
            'final_grain_sizes': self.grain_analyzer.calculate_grain_sizes(self.monte_carlo.grid),
            'final_residual_stress': statistics['residual_stress_history'][-1],
            'final_abnormal_grains': statistics['abnormal_grain_history'][-1]
        }
        
        # Add final inspection results
        final_stats['inspection_results'] = self.grain_analyzer.check_inspection_criteria(
            self.monte_carlo.grid,
            statistics['temperature_history']
        )
        
        # Update statistics
        statistics.update(final_stats)
        
        return self.monte_carlo.grid, statistics
    
    def get_inspection_results(self):
        """Get inspection results for the final microstructure.
        
        Returns:
            dict: Inspection results including pass/fail status and details
        """
        final_grid = self.monte_carlo.grid
        current_temp = self.temperature_controller.current_temperature
        
        # Calculate final grain sizes and stress
        grain_sizes = self.grain_analyzer.calculate_grain_sizes(final_grid)
        residual_stress, _ = self.grain_analyzer.calculate_residual_stress(
            final_grid, current_temp, self.material_properties
        )
        
        # Check inspection criteria
        inspection_results = self.grain_analyzer.check_inspection_criteria(
            grain_sizes, residual_stress
        )
        
        # Add detailed statistics
        inspection_results.update({
            'average_grain_size': np.mean(grain_sizes),
            'grain_size_std': np.std(grain_sizes),
            'residual_stress': residual_stress,
            'temperature': current_temp,
            'material_type': self.material_properties.material_type
        })
        
        return inspection_results 