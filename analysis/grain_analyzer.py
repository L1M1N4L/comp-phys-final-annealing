"""Grain analysis module for annealing simulation."""

import numpy as np
from scipy import stats
from scipy import ndimage

class GrainAnalyzer:
    """Class for analyzing grain microstructure."""
    
    def __init__(self, grid_size):
        """Initialize analyzer.
        
        Args:
            grid_size (int): Size of the simulation grid
        """
        self.grid_size = grid_size
        self.grain_size_threshold = 50  # Maximum allowed grain size
        self.residual_stress_threshold = 100  # Maximum allowed residual stress
        
        # Pre-allocate arrays for better performance
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.neighbors = np.array([(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)])
    
    def calculate_grain_sizes(self, grid):
        """Calculate sizes of grains in the microstructure.
        
        Args:
            grid (numpy.ndarray): Microstructure grid
            
        Returns:
            numpy.ndarray: Array of grain sizes
        """
        # Ensure grid is 2D
        if len(grid.shape) != 2:
            raise ValueError("Grid must be 2-dimensional")
        
        # Reset visited array
        self.visited.fill(False)
        
        # Use numpy operations for faster processing
        unique_grains = np.unique(grid)
        grain_sizes = []
        
        # Process each grain type
        for grain_id in unique_grains:
            # Find all cells of this grain type
            grain_mask = (grid == grain_id)
            if not np.any(grain_mask):
                continue
                
            # Get connected components
            labeled_array, num_features = ndimage.label(grain_mask)
            
            # Calculate sizes of each component
            sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
            grain_sizes.extend(sizes)
        
        return np.array(grain_sizes)
    
    def detect_abnormal_grains(self, input_data):
        """Detect abnormal grains based on size distribution.
        
        Args:
            input_data: Either a grid (numpy.ndarray) or array of grain sizes
            
        Returns:
            tuple: (number of abnormal grains, array of abnormal grain sizes)
        """
        # Handle both grid and grain sizes input
        if isinstance(input_data, np.ndarray) and len(input_data.shape) == 2:
            grain_sizes = self.calculate_grain_sizes(input_data)
        else:
            grain_sizes = np.array(input_data)
            
        if grain_sizes.size == 0:
            return 0, np.array([])
        
        # Calculate statistics using numpy
        mean_size = np.mean(grain_sizes)
        std_size = np.std(grain_sizes)
        
        # Vectorized threshold check
        threshold = mean_size + 2 * std_size
        abnormal_mask = grain_sizes > threshold
        abnormal_count = np.sum(abnormal_mask)
        abnormal_sizes = grain_sizes[abnormal_mask]
        
        return int(abnormal_count), abnormal_sizes
    
    def calculate_residual_stress(self, grid, temperature, material_properties):
        """Calculate residual stress based on thermal expansion and grain size variations.
        
        Args:
            grid (numpy.ndarray): Microstructure grid
            temperature (float): Current temperature
            material_properties (MaterialProperties): Material properties object
            
        Returns:
            tuple: (mean stress, stress field)
        """
        # Use default values if material properties not provided
        if material_properties is None:
            thermal_expansion = 12e-6
            youngs_modulus = 200e9
        else:
            thermal_expansion = material_properties.thermal_expansion
            youngs_modulus = material_properties.youngs_modulus
        
        # Calculate thermal strain
        thermal_strain = thermal_expansion * (temperature - 300)
        thermal_stress = youngs_modulus * thermal_strain
        
        # Calculate grain size variations
        grain_sizes = self.calculate_grain_sizes(grid)
        if grain_sizes.size == 0:
            return 0.0, np.zeros_like(grid)
        
        # Calculate size variations
        size_variations = np.std(grain_sizes) / np.mean(grain_sizes)
        
        # Calculate stress field
        stress_field = thermal_stress * size_variations * np.ones_like(grid)
        stress_field = ndimage.gaussian_filter(stress_field, sigma=2)
        
        return float(np.mean(stress_field)), stress_field
    
    def check_inspection_criteria(self, grid, temperature_history):
        """Check if microstructure meets inspection criteria.
        
        Args:
            grid (numpy.ndarray): Microstructure grid
            temperature_history (list): History of temperature values
            
        Returns:
            dict: Inspection results
        """
        # Calculate grain sizes
        grain_sizes = self.calculate_grain_sizes(grid)
        if grain_sizes.size == 0:
            return {
                'passed': False,
                'reason': 'No grains detected',
                'average_grain_size': 0.0,
                'abnormal_grains': 0,
                'residual_stress': 0.0
            }
        
        # Calculate statistics
        avg_grain_size = float(np.mean(grain_sizes))
        abnormal_grains, _ = self.detect_abnormal_grains(grain_sizes)
        residual_stress = self.calculate_residual_stress(grid, temperature_history[-1], None)[0]
        
        # Check criteria
        passed = True
        reasons = []
        
        if avg_grain_size > self.grain_size_threshold:
            passed = False
            reasons.append(f"Average grain size ({avg_grain_size:.2f}) exceeds threshold ({self.grain_size_threshold})")
        
        if abnormal_grains > 0:
            passed = False
            reasons.append(f"Found {abnormal_grains} abnormal grains")
        
        if residual_stress > self.residual_stress_threshold:
            passed = False
            reasons.append(f"Residual stress ({residual_stress:.2f}) exceeds threshold ({self.residual_stress_threshold})")
        
        return {
            'passed': passed,
            'reason': '; '.join(reasons) if reasons else 'All criteria met',
            'average_grain_size': avg_grain_size,
            'abnormal_grains': abnormal_grains,
            'residual_stress': residual_stress
        } 