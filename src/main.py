"""Main script for running the annealing simulation."""

import numpy as np
from tqdm import tqdm
from src.models.monte_carlo import MonteCarloSimulator
from src.models.material_properties import MaterialProperties
from src.models.temperature_controller import TemperatureController
from src.analysis.grain_analyzer import GrainAnalyzer
from src.visualization.visualizer import Visualizer

def main():
    """Run the annealing simulation."""
    print("Initializing simulation...")
    
    # Simulation parameters
    grid_size = 100
    num_grains = 50
    max_iterations = 500  # Increased for better visualization

    # Annealing process parameters
    initial_temperature = 300
    soak_temperature = 1000
    final_temperature = 300
    heating_rate = 5  # K per step (slower heating)
    soak_duration = 100  # steps at soak temp (longer soaking)
    cooling_rate = 0.5  # K per step (slower cooling)

    # Initialize components
    material_properties = MaterialProperties(material_type='steel')  # Use predefined steel properties
    
    temperature_controller = TemperatureController(
        initial_temperature=initial_temperature,
        soak_temperature=soak_temperature,
        final_temperature=final_temperature,
        heating_rate=heating_rate,
        soak_duration=soak_duration,
        cooling_rate=cooling_rate
    )
    
    simulator = MonteCarloSimulator(
        grid_size=grid_size,
        material_properties=material_properties,
        temperature_controller=temperature_controller
    )
    
    grain_analyzer = GrainAnalyzer(grid_size)
    visualizer = Visualizer(grid_size)
    
    # Create initial structure
    print("Creating initial structure...")
    simulator.create_initial_structure(num_grains)
    
    # Run simulation
    print("Running simulation...")
    simulation_data = []
    
    for iteration in tqdm(range(max_iterations), desc="Simulation Progress"):
        # Run Monte Carlo step
        simulator.monte_carlo_step(grain_analyzer)
        
        # Calculate current statistics
        current_data = {
            'grid': simulator.grid.copy(),
            'energy_history': simulator.energy_history,
            'temperature_history': simulator.temperature_history,
            'grain_size_history': simulator.grain_size_history
        }
        
        # Calculate stress field
        _, stress_field = grain_analyzer.calculate_residual_stress(
            current_data['grid'],
            simulator.temperature,
            material_properties
        )
        current_data['stress_field'] = stress_field
        
        simulation_data.append(current_data)
    
    # Save results
    print("Saving results...")
    np.save('results/simulation_data.npy', simulation_data)
    
    # Create and display animation
    print("Creating visualization...")
    anim = visualizer.create_animation(simulation_data, max_iterations)
    
    # Save animation as GIF
    print("Saving animation...")
    anim.save('results/animation.gif', writer='pillow', fps=20)
    
    # Plot final results
    print("Creating final results plot...")
    final_data = {
        'grid': simulator.grid.copy(),
        'stress_field': stress_field,
        'grain_sizes': grain_analyzer.calculate_grain_sizes(current_data['grid']),
        'energy_history': simulator.energy_history
    }
    visualizer.plot_final_results(final_data)
    
    # Print inspection results
    print("\nInspection Results:")
    inspection_results = grain_analyzer.check_inspection_criteria(
        current_data['grid'],
        simulator.temperature_history
    )
    
    for criterion, result in inspection_results.items():
        print(f"{criterion}: {'Pass' if result else 'Fail'}")

if __name__ == "__main__":
    main() 