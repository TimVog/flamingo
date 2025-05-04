import numpy as np

class CorrectionConfig:
    """Manages all correction parameters and optimization settings."""

    def __init__(self, custom_config=None):
        """
        Initialize with default parameters or custom configuration.

        Parameters:
        custom_config (dict, optional): Custom configuration parameters
        """
        # Correction enablement flags
        self.enabled_corrections = {
            "delay": True,
            "dilatation": True,
            "periodic": True
        }

        # Parameter bounds and constraints
        self.bounds = {
            "delay": {"min": -1e-11, "max": 1e-11},         # in seconds
            "dilatation": {"min": -1e-2, "max": 1e-2},      # unitless
            "residual_noise": {"min": -0.1, "max": 0.1}     # unitless
        }

        # Add explicit parameter order
        self.parameter_order = ["delay", "dilatation", "residual_noise"]

        # Periodic sampling specific parameters
        self.periodic_sampling = {
            "frequency_limit": 7.5e12,  # in Hz
            "max_iteration": 4000,
            "popsize": 16,
            "min_values": {"amplitude": 0,
                           "frequency": 6e12 * 2 * np.pi,
                           "phase": -np.pi},
            "max_values": {"amplitude": 1e-13,
                           "frequency": 12e12 * 2 * np.pi,
                           "phase": np.pi}
        }

        # Optimization settings
        self.optimization_settings = {
            "max_iteration": 5000,
            "algorithm": "SLSQP"
        }

        # Results storage
        self.results = {}

        # Apply custom configuration if provided
        if custom_config:
            self.update_from_dict(custom_config)

    def update_from_dict(self, config_dict):
        """
        Update configuration parameters from a dictionary.

        Parameters:
        config_dict (dict): Dictionary containing configuration parameters
        """
        if not isinstance(config_dict, dict):
            return

        # Update enabled corrections
        if 'enabled_corrections' in config_dict and isinstance(config_dict['enabled_corrections'], dict):
            for key, value in config_dict['enabled_corrections'].items():
                if key in self.enabled_corrections:
                    self.enabled_corrections[key] = bool(value)

        # Update bounds
        if 'bounds' in config_dict and isinstance(config_dict['bounds'], dict):
            for param, bounds in config_dict['bounds'].items():
                if param in self.bounds and isinstance(bounds, dict):
                    if 'min' in bounds and 'max' in bounds:
                        try:
                            min_val = float(bounds['min'])
                            max_val = float(bounds['max'])
                            # Ensure min <= max
                            if min_val <= max_val:
                                self.bounds[param]['min'] = min_val
                                self.bounds[param]['max'] = max_val
                        except (ValueError, TypeError):
                            pass

        # Update periodic sampling parameters
        if 'periodic_sampling' in config_dict and isinstance(config_dict['periodic_sampling'], dict):
            ps_config = config_dict['periodic_sampling']

            # Simple numeric parameters
            for key in ['frequency_limit', 'max_iteration', 'popsize']:
                if key in ps_config:
                    try:
                        if key == 'frequency_limit':
                            self.periodic_sampling[key] = float(ps_config[key])
                        else:
                            self.periodic_sampling[key] = int(ps_config[key])
                    except (ValueError, TypeError):
                        pass

            # Min/max value dictionaries
            for minmax_key in ['min_values', 'max_values']:
                if minmax_key in ps_config and isinstance(ps_config[minmax_key], dict):
                    for param, value in ps_config[minmax_key].items():
                        if param in self.periodic_sampling[minmax_key]:
                            try:
                                self.periodic_sampling[minmax_key][param] = float(value)
                            except (ValueError, TypeError):
                                pass

        # Update optimization settings
        if 'optimization_settings' in config_dict and isinstance(config_dict['optimization_settings'], dict):
            opt_config = config_dict['optimization_settings']

            if 'max_iteration' in opt_config:
                try:
                    self.optimization_settings['max_iteration'] = int(opt_config['max_iteration'])
                except (ValueError, TypeError):
                    pass

            if 'algorithm' in opt_config and opt_config['algorithm'] in ['SLSQP', 'L-BFGS-B', 'TNC']:
                self.optimization_settings['algorithm'] = opt_config['algorithm']

    def get_enabled_parameters(self):
        """Returns parameter bounds for only enabled corrections."""
        enabled_bounds = {}
        if self.enabled_corrections["delay"]:
            enabled_bounds["delay"] = self.bounds["delay"]
        if self.enabled_corrections["dilatation"]:
            enabled_bounds["dilatation"] = self.bounds["dilatation"]
        # Always include amplitude correction
        enabled_bounds["residual_noise"] = self.bounds["residual_noise"]
        return enabled_bounds

    def create_storage(self, num_traces):
        """Initialize results storage for fitted parameters."""
        self.results = {
            "metrics": {"f_opt": np.zeros(num_traces)}
        }

        # Create arrays for each parameter
        for param_name in self.get_enabled_parameters():
            self.results[param_name] = np.zeros(num_traces)

    def get_min_max_arrays(self):
        """Returns min_values and max_values arrays for optimization."""
        enabled_params = self.get_enabled_parameters()
        # Filter parameter_order to only include enabled parameters
        enabled_order = [p for p in self.parameter_order if p in enabled_params]
        min_values = np.array([enabled_params[p]["min"] for p in enabled_order])
        max_values = np.array([enabled_params[p]["max"] for p in enabled_order])
        return min_values, max_values, enabled_order

    def get_parameter_vector(self, trace_idx):
        """Returns parameter vector for a specific trace."""
        return np.array([self.results[param][trace_idx] for param in self.get_enabled_parameters()])