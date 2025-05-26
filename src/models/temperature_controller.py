"""Temperature controller for annealing simulation."""

import numpy as np

class TemperatureController:
    """Class for controlling temperature during annealing process."""
    
    def __init__(self, initial_temperature, soak_temperature, final_temperature,
                 heating_rate, soak_duration, cooling_rate):
        """Initialize temperature controller.
        
        Args:
            initial_temperature (float): Initial temperature in Kelvin
            soak_temperature (float): Soaking temperature in Kelvin
            final_temperature (float): Final temperature in Kelvin
            heating_rate (float): Heating rate in K per step
            soak_duration (int): Number of steps at soak temperature
            cooling_rate (float): Cooling rate in K per step
        """
        self.initial_temperature = initial_temperature
        self.soak_temperature = soak_temperature
        self.final_temperature = final_temperature
        self.heating_rate = heating_rate
        self.soak_duration = soak_duration
        self.cooling_rate = cooling_rate
        
        # Calculate total steps for each phase
        self.heating_steps = int((soak_temperature - initial_temperature) / heating_rate)
        self.cooling_steps = int((soak_temperature - final_temperature) / cooling_rate)
        self.total_steps = self.heating_steps + self.soak_duration + self.cooling_steps
        
        # Initialize current step and temperature
        self.current_step = 0
        self.current_temperature = initial_temperature
        
        # Create temperature profile
        self.temperature_profile = self._create_temperature_profile()
    
    def _create_temperature_profile(self):
        """Create temperature profile for the entire annealing process.
        
        Returns:
            numpy.ndarray: Array of temperatures for each step
        """
        # Create temperature profile with smooth transitions
        heating_profile = np.linspace(self.initial_temperature, self.soak_temperature, self.heating_steps)
        soak_profile = np.ones(self.soak_duration) * self.soak_temperature
        cooling_profile = np.linspace(self.soak_temperature, self.final_temperature, self.cooling_steps)
        
        # Add small random fluctuations to prevent getting stuck
        noise = np.random.normal(0, 0.01, self.total_steps)
        profile = np.concatenate([heating_profile, soak_profile, cooling_profile])
        profile = profile * (1 + noise)
        
        return profile
    
    def update_temperature(self):
        """Update temperature for the current step.
        
        Returns:
            float: Current temperature
        """
        if self.current_step < self.total_steps:
            self.current_temperature = self.temperature_profile[self.current_step]
            self.current_step += 1
        return self.current_temperature
    
    def get_phase(self):
        """Get current phase of the annealing process.
        
        Returns:
            str: Current phase ('heating', 'soaking', or 'cooling')
        """
        if self.current_step < self.heating_steps:
            return 'heating'
        elif self.current_step < self.heating_steps + self.soak_duration:
            return 'soaking'
        else:
            return 'cooling'
    
    def get_temperature_history(self):
        """Get the complete temperature history (not used in this version)."""
        return None
    
    def get_heating_rate(self):
        return self.heating_rate
    
    def get_cooling_rate(self):
        return self.cooling_rate 