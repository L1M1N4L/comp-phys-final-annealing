"""Material properties module for annealing simulation."""

class MaterialProperties:
    """Class to handle material-specific properties for annealing simulation."""
    
    # Material property definitions
    MATERIALS = {
        'aluminum': {
            'boundary_energy': 0.324,  # J/m²
            'thermal_expansion': 23.1e-6,  # 1/K
            'youngs_modulus': 70e9,  # Pa
            'recrystallization_temp': 300,  # K
            'optimal_heating_rate': 5,  # K/min
            'optimal_cooling_rate': 10,  # K/min
            'density': 2700,  # kg/m³
            'specific_heat': 900,  # J/(kg·K)
            'thermal_conductivity': 237,  # W/(m·K)
        },
        'steel': {
            'boundary_energy': 0.8,  # J/m²
            'thermal_expansion': 12e-6,  # 1/K
            'youngs_modulus': 200e9,  # Pa
            'recrystallization_temp': 600,  # K
            'optimal_heating_rate': 10,  # K/min
            'optimal_cooling_rate': 20,  # K/min
            'density': 7850,  # kg/m³
            'specific_heat': 490,  # J/(kg·K)
            'thermal_conductivity': 50,  # W/(m·K)
        }
    }
    
    def __init__(self, material_type='aluminum'):
        """Initialize material properties.
        
        Args:
            material_type (str): Type of material ('aluminum' or 'steel')
        """
        if material_type not in self.MATERIALS:
            raise ValueError(f"Unknown material type: {material_type}")
        
        self.material_type = material_type
        self.properties = self.MATERIALS[material_type]
    
    def get_property(self, property_name):
        """Get a specific material property.
        
        Args:
            property_name (str): Name of the property to retrieve
            
        Returns:
            float: The requested property value
        """
        if property_name not in self.properties:
            raise ValueError(f"Unknown property: {property_name}")
        return self.properties[property_name]
    
    @property
    def boundary_energy(self):
        """Get grain boundary energy."""
        return self.get_property('boundary_energy')
    
    @property
    def thermal_expansion(self):
        """Get thermal expansion coefficient."""
        return self.get_property('thermal_expansion')
    
    @property
    def youngs_modulus(self):
        """Get Young's modulus."""
        return self.get_property('youngs_modulus')
    
    @property
    def recrystallization_temp(self):
        """Get recrystallization temperature."""
        return self.get_property('recrystallization_temp')
    
    @property
    def optimal_heating_rate(self):
        """Get optimal heating rate."""
        return self.get_property('optimal_heating_rate')
    
    @property
    def optimal_cooling_rate(self):
        """Get optimal cooling rate."""
        return self.get_property('optimal_cooling_rate')
    
    @property
    def density(self):
        """Get material density."""
        return self.get_property('density')
    
    @property
    def specific_heat(self):
        """Get specific heat capacity."""
        return self.get_property('specific_heat')
    
    @property
    def thermal_conductivity(self):
        """Get thermal conductivity."""
        return self.get_property('thermal_conductivity') 