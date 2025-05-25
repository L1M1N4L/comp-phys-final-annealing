from simulation import AnnealingSimulation
from visualization import visualize_results

def main():
    # Create simulation with smaller values for faster execution
    sim = AnnealingSimulation(
        grid_size=50,      # Size of the simulation grid
        num_grains=20,     # Number of initial grains
        max_iterations=200 # Number of simulation steps
    )
    
    # Run the simulation
    frames = sim.run_simulation()
    
    # Create visualizations
    visualize_results(
        frames,
        sim.energy_history,
        sim.average_grain_size_history,
        sim.temperature_history
    )

if __name__ == "__main__":
    main() 