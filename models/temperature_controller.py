"""Temperature control module for annealing simulation."""

import numpy as np

class TemperatureController:
    """Class for managing temperature profiles and control during annealing."""
    
    def __init__(self, initial_temperature=1000, final_temperature=300, cooling_rate=0.1):
        """Initialize temperature controller.
        
        Args:
            initial_temperature (float): Initial temperature in Kelvin
            final_temperature (float): Final temperature in Kelvin
            cooling_rate (float): Cooling rate in Kelvin per step
        """
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.current_temperature = initial_temperature
    
    def update_temperature(self):
        """Update current temperature according to cooling rate.
        
        Returns:
            float: Current temperature
        """
        if self.current_temperature > self.final_temperature:
            self.current_temperature -= self.cooling_rate
            if self.current_temperature < self.final_temperature:
                self.current_temperature = self.final_temperature
        return self.current_temperature
    
    def get_temperature_history(self):
        """Get the complete temperature history.
        
        Returns:
            numpy.ndarray: Array of temperature values
        """
        return np.arange(self.initial_temperature, self.final_temperature, -self.cooling_rate)
    
    def get_heating_rate(self):
        """Calculate current heating rate.
        
        Returns:
            float: Current heating rate in K/min
        """
        return 0.0
    
    def get_cooling_rate(self):
        """Calculate current cooling rate.
        
        Returns:
            float: Current cooling rate in K/min
        """
        return self.cooling_rate 