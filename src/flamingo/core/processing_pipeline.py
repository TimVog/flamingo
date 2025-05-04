import numpy as np
import h5py
from scipy.signal import get_window, sosfiltfilt

# Import from other modules
from flamingo.utils import helper_functions as hf
from flamingo.utils import config
from flamingo.algorithms import error_fit_functions as ef
from flamingo.algorithms import correction_methods as cm
from flamingo.utils.progress import smartrange

class ProcessingPipeline:
    """Manages the sequence of THz data processing steps."""

    def __init__(self, correction_config):
        """
        Initialize pipeline with configuration.

        Parameters:
        correction_config (CorrectionConfig): Parameter configuration object
        """
        self.config = correction_config
        self.data = None
        self.trace_time = None
        self.freq = None
        self.filter_coefficients = None
        self.window = None
        self.dt = None
        self.reference_trace = None

        # Store current correction configuration for change detection
        self.previous_config = {k: v for k, v in self.config.enabled_corrections.items()}

    def process_file(self, filepath, trace_start, trace_end, lowcut):
        """
        Process THz data file with the configured correction steps.
        """
        if trace_end <= trace_start:
            raise ValueError("trace_end must be greater than trace_start")
        if lowcut <= 0:
            raise ValueError("lowcut must be positive")

        # Load basic data and prepare structures
        with h5py.File(filepath, "r") as input_file:
            # Step 0: Initialize data structures
            self._initialize(input_file, trace_start, trace_end, lowcut)

            # Step 1: Calculate statistics
            config.logger.info("Step 1/5: Determining mean and standard deviation...")
            self._calculate_mean_std(input_file, trace_start, trace_end)

            # Step 2: Find reference trace
            config.logger.info("Step 2/5: Finding reference trace...")
            self._find_reference_trace(input_file, trace_start, trace_end)

            # Step 3: Fit periodic sampling (if enabled)
            if self.config.enabled_corrections["periodic"]:
                config.logger.info("Step 3/5: Fitting periodic sampling error...")
                self._fit_periodic_sampling()
            else:
                config.logger.info("Step 3/5: Skipping periodic sampling error (disabled)")

            # Step 4: Initialize optimization parameters
            config.logger.info("Step 4/5: Initializing optimization parameters...")
            min_values, max_values, param_order = self._initialize_optimization(input_file, trace_start)

            # Step 5: Apply corrections to all traces
            config.logger.info("Step 5/5: Applying corrections to all traces...")
            self._apply_corrections(input_file, trace_start, trace_end, min_values, max_values, param_order)

        # Store current configuration for next run
        self.previous_config = {k: v for k, v in self.config.enabled_corrections.items()}

        return self.data, self.config.results, self.trace_time, self.freq

    def _initialize(self, input_file, trace_start, trace_end, lowcut):
        """Initialize data structures and preprocessing elements."""
        # Read time axis
        self.trace_time = np.array(input_file["timeaxis"]) * 1e-12

        # Setup metadata
        number_of_traces = len(range(trace_start, trace_end))
        number_of_samples = len(self.trace_time)
        config.logger.info(f"Detected {number_of_traces} traces, each {number_of_samples} samples long.")

        # Calculate time step and frequency axis
        self.dt = np.mean(np.diff(self.trace_time))
        self.freq = np.fft.rfftfreq(number_of_samples, self.dt)

        # Initialize data dictionary with correct data structure names based on active corrections
        self.data = self._create_data_dict_with_proper_labels(self.trace_time)

        # Create window function
        self.window = get_window(("tukey", 0.05), len(self.trace_time), fftbins=False)

        # Calculate filter coefficients
        self.filter_coefficients = hf.get_filter_coefficients(fs=1/self.dt, lowcut=lowcut, highcut=None, order=5)

        # Initialize results storage
        self.config.create_storage(number_of_traces)

    def _create_data_dict_with_proper_labels(self, trace_time):
        """Create data dictionary with proper labels based on active corrections."""
        trace_time_fft = np.fft.rfft(trace_time)
        data = {}

        # Common processing steps
        processing_steps = ["raw", "filtered_windowed"]

        # Add proper correction labels based on active corrections
        if self.config.enabled_corrections["dilatation"]:
            processing_steps.append("corrected (delay_dilatation)")
        else:
            processing_steps.append("corrected (delay)")

        # Add periodic correction if enabled
        if self.config.enabled_corrections["periodic"]:
            if self.config.enabled_corrections["dilatation"]:
                processing_steps.append("corrected (+periodic)")
            else:
                processing_steps.append("corrected (delay+periodic)")

        # Create the data structure with the appropriate labels
        for process_step in processing_steps:
            data[process_step] = {}
            for domain in ["time", "freq"]:
                data[process_step][domain] = {
                    "mean": np.zeros(len(trace_time) if domain == "time" else len(trace_time_fft)),
                    "std": np.zeros(len(trace_time) if domain == "time" else len(trace_time_fft))}

        return data

    def _calculate_mean_std(self, input_file, trace_start, trace_end):
        """Calculate mean and standard deviation for raw and filtered data."""
        number_of_traces = len(range(trace_start, trace_end))

        for count, i in enumerate(smartrange(trace_start, trace_end,
                                             desc="Step 1/5: Statistics")):
            # Read raw data
            trace_data = np.array(input_file[str(i)])

            # Calculate stats for raw data
            self.data = hf.calculate_mean_std(trace_data, count, "raw", "time", self.data)
            trace_data_fft = np.fft.rfft(trace_data)
            self.data = hf.calculate_mean_std(trace_data_fft, count, "raw", "freq", self.data)

            # Filter and window data
            trace_filtered = sosfiltfilt(self.filter_coefficients, trace_data, padtype=None)
            trace_windowed = trace_filtered * self.window

            # Calculate stats for filtered data
            self.data = hf.calculate_mean_std(trace_windowed, count, "filtered_windowed", "time", self.data)
            trace_windowed_fft = np.fft.rfft(trace_windowed)
            self.data = hf.calculate_mean_std(trace_windowed_fft, count, "filtered_windowed", "freq", self.data)

        # Normalize statistics
        self.data = hf.normalize_std(number_of_traces, self.data)

    def _find_reference_trace(self, input_file, trace_start, trace_end):
        """Find reference trace closest to mean trace for optimization."""
        # Normalize mean trace for cosine similarity calculation
        trace_mean = self.data["filtered_windowed"]["time"]["mean"]
        trace_mean_normed = trace_mean / np.linalg.norm(trace_mean)

        # Find closest trace to mean
        min_distance = np.inf
        self.reference_trace = None
        reference_idx = None

        for i in smartrange(trace_start, trace_end,
                            desc="Step 2/5: Reference trace"):
            # Process trace
            trace_data = np.array(input_file[str(i)])
            trace_filtered = sosfiltfilt(self.filter_coefficients, trace_data, padtype=None)
            trace_windowed = trace_filtered * self.window

            # Calculate similarity
            trace_normed = trace_windowed / np.linalg.norm(trace_windowed)
            distance = hf.find_reference_trace(trace_mean_normed, trace_normed)

            # Update reference if closer to mean
            if distance < min_distance:
                min_distance = distance
                self.reference_trace = trace_windowed
                reference_idx = i

        config.logger.info(f"Found reference trace with id {reference_idx}")

    def _fit_periodic_sampling(self):
        """Fit periodic sampling error to mean trace."""
        # Use periodic_sampling parameters from config
        periodic_params = ef.fit_periodic_sampling(
            self.config.periodic_sampling,
            self.trace_time,
            self.data["filtered_windowed"]["time"]["mean"],
            self.freq,
            self.dt
        )

        # Store parameters for later application
        self.config.results["periodic_params"] = periodic_params

        # Pre-calculate cosine term for efficiency
        self.periodic_correction_term = periodic_params[0] * np.cos(
            periodic_params[1] * self.trace_time + periodic_params[2]
        )

    def _initialize_optimization(self, input_file, trace_start):
        """Initialize optimization parameters (Step 4/5)"""
        # Get min/max values
        min_values, max_values, param_order = self.config.get_min_max_arrays()

        # Check if configuration changed since last run
        config_changed = False
        if hasattr(self, 'previous_config'):
            for key, value in self.config.enabled_corrections.items():
                if key in self.previous_config and self.previous_config[key] != value:
                    config_changed = True
                    config.logger.info(f"Correction option changed: {key}")
                    break

        # Force reinitialization if configuration changed
        if config_changed and hasattr(self, 'x0'):
            config.logger.info("Correction options changed - reinitializing parameters")
            # Clear previous parameters to force reinitialization
            delattr(self, 'x0')

        # Read first trace
        trace_data = np.array(input_file[str(trace_start)])
        trace_filtered = sosfiltfilt(self.filter_coefficients, trace_data, padtype=None)
        trace_windowed = trace_filtered * self.window
        trace_windowed_fft = np.fft.rfft(trace_windowed)

        # Initialize parameters without duplicate logging
        optimization_params = ef.initialize_delay_dilatation(
            {"max_iteration": self.config.optimization_settings["max_iteration"],
             "min_values": min_values,
             "max_values": max_values,
             "param_order": param_order},
            trace_windowed_fft,
            self.freq,
            self.trace_time,
            self.reference_trace,
            self.dt
        )

        # Store initial parameters
        self.x0 = optimization_params["x0"]

        return min_values, max_values, param_order

    def _apply_corrections(self, input_file, trace_start, trace_end, min_values, max_values, param_order):
        """Apply corrections to all traces (Step 5/5)"""
        # Get the initial optimization parameters
        x0 = self.x0

        # Determine correct data dictionary keys based on enabled corrections
        delay_key = "corrected (delay_dilatation)" if self.config.enabled_corrections["dilatation"] else "corrected (delay)"
        periodic_key = "corrected (+periodic)" if self.config.enabled_corrections["dilatation"] else "corrected (delay+periodic)"

        # Process each trace with clear progress description
        for count, i in enumerate(smartrange(trace_start, trace_end,
                                             desc="Step 5/5: Applying corrections")):
            try:
                # Process trace data
                trace_data = np.array(input_file[str(i)])
                trace_filtered = sosfiltfilt(self.filter_coefficients, trace_data, padtype=None)
                trace_windowed = trace_filtered * self.window
                trace_windowed_fft = np.fft.rfft(trace_windowed)

                # Fit parameters using the existing index approach
                result = ef.fit_delay_dilatation(
                    {"x0": x0, "max_iteration": self.config.optimization_settings["max_iteration"],
                     "min_values": min_values, "max_values": max_values, "param_order": param_order},
                    self.config.results,
                    trace_windowed_fft,
                    self.freq,
                    self.trace_time,
                    self.reference_trace,
                    self.dt,
                    count
                )

                # Update optimization parameters for next iteration
                x0 = result[0]["x0"]

                # Create parameter dictionary for correction application
                correction_params = {}
                for param in param_order:
                    if param in self.config.results and count < len(self.config.results[param]):
                        correction_params[param] = self.config.results[param][count]

                # Add periodic params if available
                if "periodic_params" in self.config.results:
                    correction_params["periodic_params"] = self.config.results["periodic_params"]

                # Apply delay and dilatation corrections
                trace_corrected = cm.apply_delay_dilatation_correction(
                    trace_windowed_fft,
                    self.trace_time,
                    self.freq,
                    self.dt,
                    correction_params,
                    self.config.enabled_corrections
                )

                # Calculate statistics for corrected trace (delay+dilatation)
                self.data = hf.calculate_mean_std(
                    trace_corrected,
                    count,
                    delay_key,
                    "time",
                    self.data
                )
                trace_corrected_fft = np.fft.rfft(trace_corrected)
                self.data = hf.calculate_mean_std(
                    trace_corrected_fft,
                    count,
                    delay_key,
                    "freq",
                    self.data
                )

                # Apply periodic correction if enabled
                if self.config.enabled_corrections["periodic"]:
                    trace_periodic = trace_corrected - np.gradient(trace_corrected, self.dt) * self.periodic_correction_term

                    # Calculate statistics for periodic correction
                    self.data = hf.calculate_mean_std(
                        trace_periodic,
                        count,
                        periodic_key,
                        "time",
                        self.data
                    )
                    trace_periodic_fft = np.fft.rfft(trace_periodic)
                    self.data = hf.calculate_mean_std(
                        trace_periodic_fft,
                        count,
                        periodic_key,
                        "freq",
                        self.data
                    )
            except Exception as e:
                config.logger.error(f"Error processing trace {i}: {str(e)}")
                # Continue with next trace instead of stopping the entire processing
                continue

        # Normalize statistics for corrected data
        process_steps = [delay_key]
        if self.config.enabled_corrections["periodic"]:
            process_steps.append(periodic_key)

        self.data = hf.normalize_std(
            len(range(trace_start, trace_end)),
            self.data,
            process_step=process_steps
        )

    def export_corrected_data(self, input_filepath, output_filepath, trace_start, trace_end):
        """Export corrected data to a new HDF5 file."""
        config.logger.info(f"Exporting corrected data to {output_filepath}...")

        # Open input and output files
        with h5py.File(input_filepath, "r") as input_file, h5py.File(output_filepath, "w") as output_file:
            # Copy time axis and ensure proper initialization
            trace_time = np.array(input_file["timeaxis"][:]) * 1e-12
            output_file.create_dataset("timeaxis", data=input_file["timeaxis"][:])

            # Calculate time step
            dt = np.mean(np.diff(trace_time))

            # Ensure all required parameters are initialized
            if not hasattr(self, 'trace_time') or self.trace_time is None:
                self.trace_time = trace_time

            if not hasattr(self, 'dt') or self.dt is None:
                self.dt = dt

            if not hasattr(self, 'freq') or self.freq is None:
                # Calculate frequency axis based on time data
                self.freq = np.fft.rfftfreq(len(trace_time), dt)

            # Recalculate filter coefficients
            self.filter_coefficients = hf.get_filter_coefficients(
                fs=1/dt, lowcut=0.2e12, highcut=None, order=5)

            # Create window function
            self.window = get_window(("tukey", 0.05), len(trace_time), fftbins=False)

            # Process each trace with robust parameter handling
            for i in smartrange(trace_start, trace_end, desc="Exporting traces"):
                try:
                    # Read and process trace data...
                    trace_data = np.array(input_file[str(i)])
                    trace_filtered = sosfiltfilt(self.filter_coefficients, trace_data, padtype=None)
                    trace_windowed = trace_filtered * self.window
                    trace_windowed_fft = np.fft.rfft(trace_windowed)

                    # Safe parameter access with defaults
                    idx = i - trace_start
                    correction_params = {
                        "delay": self.config.results.get("delay", [0])[idx] if idx < len(self.config.results.get("delay", [])) else 0,
                        "dilatation": self.config.results.get("dilatation", [0])[idx] if idx < len(self.config.results.get("dilatation", [])) else 0,
                        "residual_noise": self.config.results.get("residual_noise", [0])[idx] if idx < len(self.config.results.get("residual_noise", [])) else 0
                    }

                    # Apply corrections with safety checks
                    corrected_trace = cm.apply_delay_dilatation_correction(
                        trace_windowed_fft,
                        self.trace_time,
                        self.freq,
                        self.dt,
                        correction_params,
                        self.config.enabled_corrections
                    )

                    # Save corrected trace
                    output_file.create_dataset(str(i), data=corrected_trace)

                except Exception as e:
                    config.logger.error(f"Error processing trace {i}: {str(e)}")
                    # Continue with next trace instead of stopping the entire export
                    continue

            config.logger.info(f"Successfully exported corrected traces.")