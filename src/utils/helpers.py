"""Utility functions for the annealing simulation."""

import numpy as np
from typing import Tuple, List, Dict, Any
import json
import os
from datetime import datetime

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_simulation_results(grid: np.ndarray, statistics: Dict[str, Any], 
                          output_dir: str = 'results') -> None:
    """Save simulation results to files.
    
    Args:
        grid (np.ndarray): Final microstructure grid
        statistics (dict): Simulation statistics
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save grid
    np.save(os.path.join(output_dir, f'final_grid_{timestamp}.npy'), grid)
    
    # Convert NumPy types to Python native types for JSON serialization
    serializable_stats = convert_numpy_types(statistics)
    
    # Save statistics
    with open(os.path.join(output_dir, f'statistics_{timestamp}.json'), 'w') as f:
        json.dump(serializable_stats, f, indent=4)
    
    print(f"Results saved to {os.path.join(output_dir, f'final_grid_{timestamp}.npy')} and {os.path.join(output_dir, f'statistics_{timestamp}.json')}")

def load_simulation_results(output_dir: str = 'results') -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load simulation results from files.
    
    Args:
        output_dir (str): Directory containing results
        
    Returns:
        tuple: (grid, statistics)
    """
    # Load grid
    grid = np.load(os.path.join(output_dir, 'final_grid.npy'))
    
    # Load statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'r') as f:
        statistics = json.load(f)
    
    return grid, statistics

def calculate_grain_statistics(grain_sizes: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for grain sizes.
    
    Args:
        grain_sizes (list): List of grain sizes
        
    Returns:
        dict: Statistical measures
    """
    return {
        'mean': np.mean(grain_sizes),
        'std': np.std(grain_sizes),
        'min': np.min(grain_sizes),
        'max': np.max(grain_sizes),
        'median': np.median(grain_sizes),
        'skewness': scipy.stats.skew(grain_sizes),
        'kurtosis': scipy.stats.kurtosis(grain_sizes)
    }

def validate_simulation_parameters(grid_size: int, num_grains: int, 
                                 max_iterations: int) -> bool:
    """Validate simulation parameters.
    
    Args:
        grid_size (int): Size of the simulation grid
        num_grains (int): Number of initial grains
        max_iterations (int): Maximum number of Monte Carlo steps
        
    Returns:
        bool: True if parameters are valid
    """
    if grid_size <= 0 or num_grains <= 0 or max_iterations <= 0:
        return False
    
    if num_grains > grid_size * grid_size:
        return False
    
    return True

def format_inspection_results(results: Dict[str, Any]) -> str:
    """Format inspection results for display.
    
    Args:
        results (dict): Inspection results
        
    Returns:
        str: Formatted results string
    """
    output = []
    output.append("Inspection Results:")
    output.append("-" * 50)
    
    # Add pass/fail status
    status = "PASS" if results.get('pass', False) else "FAIL"
    output.append(f"Status: {status}")
    
    # Add detailed results
    output.append("\nDetailed Results:")
    for key, value in results.items():
        if key != 'pass':
            if isinstance(value, float):
                output.append(f"{key}: {value:.3f}")
            else:
                output.append(f"{key}: {value}")
    
    return "\n".join(output) 