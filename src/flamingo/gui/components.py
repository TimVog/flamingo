import tkinter as tk
import customtkinter as ctk
from tkinter import TclError
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from flamingo.utils.config import logger

class OptimizationPanel:
    """Panel for configuring optimization parameters and bounds."""

    def __init__(self, parent):
        """Initialize optimization parameters panel."""
        self.parent = parent

        # Create primary frame
        self.frame = ctk.CTkFrame(parent)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create scrollable frame for parameters
        self.scrollable_frame = ctk.CTkScrollableFrame(self.frame)
        self.scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Initialize parameter variables
        self.initialize_variables()

        # Set up the form elements
        self._setup_ui()

    def initialize_variables(self):
        """Initialize variables for all optimization parameters."""
        # Correction bounds
        self.delay_min = tk.StringVar(value="-1e-11")
        self.delay_max = tk.StringVar(value="1e-11")
        self.dilatation_min = tk.StringVar(value="-1e-2")
        self.dilatation_max = tk.StringVar(value="1e-2")
        self.residual_noise_min = tk.StringVar(value="-0.1")
        self.residual_noise_max = tk.StringVar(value="0.1")

        # Periodic sampling parameters
        self.frequency_limit = tk.StringVar(value="7.5e12")
        self.max_iteration_periodic = tk.StringVar(value="4000")
        self.popsize = tk.StringVar(value="16")
        self.amplitude_min = tk.StringVar(value="0")
        self.amplitude_max = tk.StringVar(value="1e-13")
        # Use Hz values directly
        self.frequency_min = tk.StringVar(value="6.0e12")
        self.frequency_max = tk.StringVar(value="12.0e12")
        self.phase_min = tk.StringVar(value=str(-3.14159))
        self.phase_max = tk.StringVar(value=str(3.14159))

        # Optimization settings
        self.max_iteration = tk.StringVar(value="5000")
        self.algorithm = tk.StringVar(value="SLSQP")

    def _setup_ui(self):
        """Create and arrange UI elements for optimization parameters."""
        row = 0

        # Title
        title_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="Optimization Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=row, column=0, padx=10, pady=(10, 20))
        row += 1

        # --- Correction Bounds Section ---
        self._add_section_header("Correction Bounds", row)
        row += 1

        # Delay bounds
        row = self._add_min_max_field("Delay (s)", self.delay_min, self.delay_max, row, tooltip="Time delay correction bounds in seconds")

        # Dilatation bounds
        row = self._add_min_max_field("Dilatation", self.dilatation_min, self.dilatation_max, row, tooltip="Dilatation correction bounds (unitless)")

        # Residual noise bounds
        row = self._add_min_max_field("Residual Noise", self.residual_noise_min, self.residual_noise_max, row, tooltip="Residual noise correction bounds (unitless)")

        # --- Periodic Sampling Section ---
        self._add_section_header("Periodic Sampling Parameters", row)
        row += 1

        # Max iteration for periodic fitting
        row = self._add_single_field("Max Iterations", self.max_iteration_periodic, row, tooltip="Maximum iterations for periodic sampling optimization")

        # Population size
        row = self._add_single_field("Population Size", self.popsize, row, tooltip="Population size for differential evolution algorithm")

        # Frequency limit
        row = self._add_single_field("Frequency Limit (Hz)", self.frequency_limit, row, tooltip="Upper frequency limit for periodic error correction")

        # Frequency bounds
        row = self._add_min_max_field_with_suffix("Frequency (Hz)", self.frequency_min, self.frequency_max, "× 2π", row,
                                                  tooltip="Frequency bounds for periodic sampling in Hz × 2π")

        # Amplitude bounds
        row = self._add_min_max_field("Amplitude", self.amplitude_min, self.amplitude_max, row, tooltip="Amplitude bounds for periodic sampling")

        # Phase bounds
        row = self._add_min_max_field("Phase (rad)", self.phase_min, self.phase_max, row, tooltip="Phase bounds for periodic sampling in radians")

        # --- Optimization Settings Section ---
        self._add_section_header("Optimization Settings", row)
        row += 1

        # Max iteration
        row = self._add_single_field("Max Iterations", self.max_iteration, row, tooltip="Maximum iterations for main optimization algorithm")

        # Algorithm selection
        algorithm_label = ctk.CTkLabel(self.scrollable_frame, text="Algorithm")
        algorithm_label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
        row += 1

        # Algorithm dropdown
        algorithm_dropdown = ctk.CTkOptionMenu(
            self.scrollable_frame,
            values=["SLSQP", "L-BFGS-B", "TNC"],
            variable=self.algorithm,
            width=200  # Standardized width
        )
        algorithm_dropdown.grid(row=row, column=0, padx=10, pady=(0, 10), sticky="")
        row += 1

        # Reset to defaults button
        reset_button = ctk.CTkButton(
            self.scrollable_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults,
            width=200  # Standardized width
        )
        reset_button.grid(row=row, column=0, padx=10, pady=(20, 10), sticky="")

        # Add validation bindings to all entries
        self._add_validation_to_entries()

    def _add_section_header(self, text, row):
        """Add a section header with specified text."""
        section_label = ctk.CTkLabel(
            self.scrollable_frame,
            text=text,
            font=ctk.CTkFont(weight="bold")
        )
        section_label.grid(row=row, column=0, padx=10, pady=(20, 5), sticky="w")
        return row + 1

    def _add_single_field(self, label_text, variable, row, tooltip=None):
        """Add a labeled single entry field."""
        # Label
        label = ctk.CTkLabel(self.scrollable_frame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
        row += 1

        # Entry with standardized width
        entry = ctk.CTkEntry(self.scrollable_frame, textvariable=variable, width=200)
        entry.grid(row=row, column=0, padx=10, pady=(0, 5), sticky="")

        return row + 1

    def _add_min_max_field(self, label_text, min_var, max_var, row, tooltip=None):
        """Add min/max entry fields with a label."""
        # Label
        label = ctk.CTkLabel(self.scrollable_frame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
        row += 1

        # Frame for min/max entries with standardized width
        minmax_frame = ctk.CTkFrame(self.scrollable_frame)
        minmax_frame.grid(row=row, column=0, padx=10, pady=(0, 5), sticky="")

        # Min label and entry
        min_label = ctk.CTkLabel(minmax_frame, text="Min:", width=50)
        min_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        min_entry = ctk.CTkEntry(minmax_frame, textvariable=min_var, width=150)
        min_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Max label and entry
        max_label = ctk.CTkLabel(minmax_frame, text="Max:", width=50)
        max_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        max_entry = ctk.CTkEntry(minmax_frame, textvariable=max_var, width=150)
        max_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        return row + 1

    def _add_min_max_field_with_suffix(self, label_text, min_var, max_var, suffix, row, tooltip=None):
        """Add min/max entry fields with a label and suffix text."""
        # Label
        label = ctk.CTkLabel(self.scrollable_frame, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=(10, 0), sticky="w")
        row += 1

        # Frame for min/max entries with standardized width
        minmax_frame = ctk.CTkFrame(self.scrollable_frame)
        minmax_frame.grid(row=row, column=0, padx=10, pady=(0, 5), sticky="")

        # Min label, entry and suffix
        min_label = ctk.CTkLabel(minmax_frame, text="Min:", width=50)
        min_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        min_entry = ctk.CTkEntry(minmax_frame, textvariable=min_var, width=150)
        min_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        min_suffix = ctk.CTkLabel(minmax_frame, text=suffix)
        min_suffix.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Max label, entry and suffix
        max_label = ctk.CTkLabel(minmax_frame, text="Max:", width=50)
        max_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        max_entry = ctk.CTkEntry(minmax_frame, textvariable=max_var, width=150)
        max_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        max_suffix = ctk.CTkLabel(minmax_frame, text=suffix)
        max_suffix.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        return row + 1

    def _add_validation_to_entries(self):
        """Add validation to all entry fields."""
        # This would add validation callbacks to each entry
        # For scientific notation and numeric validation
        pass

    def _validate_scientific_notation(self, widget, variable):
        """Validate scientific notation input."""
        def validate_callback(*args):
            try:
                current = widget.get()
                if current == "":
                    return

                # Try to convert to float to validate
                value = float(current)
                variable.set(current)
            except (ValueError, TclError):
                widget.delete(0, "end")
                widget.insert(0, variable.get())
        return validate_callback

    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        # Reset correction bounds
        self.delay_min.set("-1e-11")
        self.delay_max.set("1e-11")
        self.dilatation_min.set("-1e-2")
        self.dilatation_max.set("1e-2")
        self.residual_noise_min.set("-0.1")
        self.residual_noise_max.set("0.1")

        # Reset periodic sampling parameters
        self.frequency_limit.set("7.5e12")
        self.max_iteration_periodic.set("4000")
        self.popsize.set("16")
        self.amplitude_min.set("0")
        self.amplitude_max.set("1e-13")
        # Reset frequencies in Hz
        self.frequency_min.set("6.0e12")
        self.frequency_max.set("12.0e12")
        self.phase_min.set(str(-3.14159))
        self.phase_max.set(str(3.14159))

        # Reset optimization settings
        self.max_iteration.set("5000")
        self.algorithm.set("SLSQP")

    def get_optimization_parameters(self):
        """Get all optimization parameters as a dictionary."""
        try:
            # Get frequency values and multiply by 2π where needed
            freq_min = float(self.frequency_min.get()) * 2 * np.pi
            freq_max = float(self.frequency_max.get()) * 2 * np.pi

            # Parse all string values to appropriate numeric types
            params = {
                # Correction bounds
                "bounds": {
                    "delay": {"min": float(self.delay_min.get()), "max": float(self.delay_max.get())},
                    "dilatation": {"min": float(self.dilatation_min.get()), "max": float(self.dilatation_max.get())},
                    "residual_noise": {"min": float(self.residual_noise_min.get()), "max": float(self.residual_noise_max.get())}
                },
                # Periodic sampling
                "periodic_sampling": {
                    "frequency_limit": float(self.frequency_limit.get()),
                    "max_iteration": int(float(self.max_iteration_periodic.get())),
                    "popsize": int(float(self.popsize.get())),
                    "min_values": {
                        "amplitude": float(self.amplitude_min.get()),
                        "frequency": freq_min,  # Multiply by 2π
                        "phase": float(self.phase_min.get())
                    },
                    "max_values": {
                        "amplitude": float(self.amplitude_max.get()),
                        "frequency": freq_max,  # Multiply by 2π
                        "phase": float(self.phase_max.get())
                    }
                },
                # Optimization settings
                "optimization_settings": {
                    "max_iteration": int(float(self.max_iteration.get())),
                    "algorithm": self.algorithm.get()
                }
            }
            return params
        except ValueError as e:
            # Handle conversion errors
            logger.error(f"Error parsing optimization parameters: {str(e)}")
            return None

class ControlPanel:
    """Sidebar panel with processing controls and parameters."""

    def __init__(self, parent, callbacks):
        """Initialize control panel with callbacks for UI events."""
        self.parent = parent
        self.callbacks = callbacks

        # Use StringVar for all numeric fields
        self.file_path = tk.StringVar(value="No file selected")
        self.trace_start = tk.StringVar(value="0")
        self.trace_end = tk.StringVar(value="1000")
        self.lowcut = tk.StringVar(value="0.2e12")
        self.enable_dilatation = tk.BooleanVar(value=True)
        self.enable_periodic = tk.BooleanVar(value=True)

        # Create frame
        self.frame = ctk.CTkFrame(parent, width=200)
        self.frame.grid_columnconfigure(0, weight=1)

        # Setup UI components
        self._setup_ui()

    def _validate_numeric_input(self, widget, variable):
        """Validate numeric input and handle empty fields."""
        def validate_callback(*args):
            try:
                current = widget.get()
                if current == "":
                    return

                # For trace indices (start/end), format as clean integers
                if variable in (self.trace_start, self.trace_end):
                    value = int(float(current))
                    widget.delete(0, "end")
                    widget.insert(0, f"{value}")  # Format without decimal
                else:
                    # For floating values like lowcut
                    value = float(current)
                    variable.set(current)
            except (ValueError, TclError):
                widget.delete(0, "end")
                widget.insert(0, variable.get())
        return validate_callback

    def _setup_ui(self):
        """Create and arrange UI elements."""
        # Title label
        title_label = ctk.CTkLabel(
            self.frame,
            text="Processing Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=10, pady=(10, 20))

        # File selection
        file_label = ctk.CTkLabel(self.frame, text="Input HDF5 File:")
        file_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")

        self.file_path_label = ctk.CTkLabel(
            self.frame,
            textvariable=self.file_path,
            wraplength=180
        )
        self.file_path_label.grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

        browse_button = ctk.CTkButton(
            self.frame,
            text="Browse...",
            command=self._on_browse
        )
        browse_button.grid(row=3, column=0, padx=10, pady=(0, 15))

        # Trace range selection
        range_frame = ctk.CTkFrame(self.frame)
        range_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        range_frame.grid_columnconfigure(0, weight=1)
        range_frame.grid_columnconfigure(1, weight=1)

        start_label = ctk.CTkLabel(range_frame, text="Start Trace:")
        start_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        start_entry = ctk.CTkEntry(range_frame, textvariable=self.trace_start, width=70)
        start_entry.grid(row=0, column=1, padx=5, pady=5)
        start_entry.bind("<FocusOut>", self._validate_numeric_input(start_entry, self.trace_start))

        # Add the missing "End Trace" label
        end_label = ctk.CTkLabel(range_frame, text="End Trace:")
        end_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        end_entry = ctk.CTkEntry(range_frame, textvariable=self.trace_end, width=70)
        end_entry.grid(row=1, column=1, padx=5, pady=5)
        end_entry.bind("<FocusOut>", self._validate_numeric_input(end_entry, self.trace_end))

        # Low cutoff frequency
        lowcut_label = ctk.CTkLabel(self.frame, text="Low Cut Frequency (Hz)")
        lowcut_label.grid(row=5, column=0, padx=10, pady=(15, 0), sticky="w")

        lowcut_entry = ctk.CTkEntry(self.frame, textvariable=self.lowcut, width=200)
        lowcut_entry.grid(row=6, column=0, padx=10, pady=(0, 15), sticky="")
        lowcut_entry.bind("<FocusOut>", self._validate_numeric_input(lowcut_entry, self.lowcut))

        # Correction options
        options_label = ctk.CTkLabel(
            self.frame,
            text="Correction Options:",
            font=ctk.CTkFont(weight="bold")
        )
        options_label.grid(row=7, column=0, padx=10, pady=(10, 5), sticky="w")

        dilatation_check = ctk.CTkCheckBox(
            self.frame,
            text="Enable Dilatation Correction",
            variable=self.enable_dilatation
        )
        dilatation_check.grid(row=8, column=0, padx=10, pady=5, sticky="w")

        periodic_check = ctk.CTkCheckBox(
            self.frame,
            text="Enable Periodic Correction",
            variable=self.enable_periodic
        )
        periodic_check.grid(row=9, column=0, padx=10, pady=5, sticky="w")

        # Processing button
        self.process_button = ctk.CTkButton(
            self.frame,
            text="Process Data",
            command=self._on_process,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.process_button.grid(row=10, column=0, padx=10, pady=(20, 5), sticky="ew")

        # Export button (initially disabled)
        self.export_button = ctk.CTkButton(
            self.frame,
            text="Export Corrected Data",
            command=self._on_export,
            state="disabled"
        )
        self.export_button.grid(row=11, column=0, padx=10, pady=5, sticky="ew")

        # Status indicator label
        self.status_label = ctk.CTkLabel(
            self.frame,
            text="Ready to process data",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=12, column=0, padx=10, pady=(20, 10))

    def _on_browse(self):
        """Handle browse button click."""
        if 'browse' in self.callbacks:
            self.callbacks['browse'](self)

    def _on_process(self):
        """Handle process button click."""
        if 'process' in self.callbacks:
            self.callbacks['process'](self)

    def _on_export(self):
        """Handle export button click."""
        if 'export' in self.callbacks:
            self.callbacks['export'](self)

    def get_parameters(self):
        """Return current processing parameters."""
        try:
            trace_start = int(float(self.trace_start.get())) if self.trace_start.get() else 0
            trace_end = int(float(self.trace_end.get())) if self.trace_end.get() else 1000
            lowcut = float(self.lowcut.get()) if self.lowcut.get() else 0.2e12
        except ValueError:
            # Handle invalid values with defaults
            trace_start = 0
            trace_end = 1000
            lowcut = 0.2e12

        return {
            'file_path': self.file_path.get(),
            'trace_start': trace_start,
            'trace_end': trace_end,
            'lowcut': lowcut,
            'enable_dilatation': self.enable_dilatation.get(),
            'enable_periodic': self.enable_periodic.get()
        }

    def set_file_path(self, path):
        """Update file path."""
        self.file_path.set(path)

    def set_trace_range(self, start, end):
        """Update trace range."""
        # Format integers without decimal point
        self.trace_start.set(f"{int(start)}")  # Force integer formatting
        self.trace_end.set(f"{int(end)}")      # Force integer formatting

    def set_export_enabled(self, enabled):
        """Enable/disable export button."""
        self.export_button.configure(state="normal" if enabled else "disabled")

    def set_status(self, status_text):
        """Update status label."""
        self.status_label.configure(text=status_text)

    def set_processing_in_progress(self, in_progress):
        """Update UI for processing state."""
        state = "disabled" if in_progress else "normal"
        self.process_button.configure(state=state)

class VisualizationPanel:
    """Tabbed visualization panel for displaying plots."""

    def __init__(self, parent):
        """Initialize visualization panel with tabs for different plots."""
        self.parent = parent

        # Create main frame
        self.frame = ctk.CTkFrame(parent)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        # Create tabview for visualizations
        self.tabview = ctk.CTkTabview(self.frame)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Add tabs for different plot types
        self.tab_processing = self.tabview.add("Processing Steps")
        self.tab_comparison = self.tabview.add("Comparison")
        self.tab_parameters = self.tabview.add("Correction Parameters")
        self.tab_optimization = self.tabview.add("Optimization Settings")  # New tab

        # Configure tab layouts
        for tab in [self.tab_processing, self.tab_comparison, self.tab_parameters]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        # Special configuration for optimization tab
        self.tab_optimization.grid_columnconfigure(0, weight=1)
        self.tab_optimization.grid_rowconfigure(0, weight=1)

        # Initialize matplotlib canvases
        self._setup_matplotlib_canvas(self.tab_processing, "processing")
        self._setup_matplotlib_canvas(self.tab_comparison, "comparison")
        self._setup_matplotlib_canvas(self.tab_parameters, "parameters")

        # Initialize optimization panel
        self.optimization_panel = OptimizationPanel(self.tab_optimization)
        self.optimization_panel.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Add initial guidance text
        self._add_initial_guidance()

    def _setup_matplotlib_canvas(self, parent, name_prefix):
        """Set up matplotlib canvas in the given parent widget."""
        # Create frame for canvas
        canvas_frame = ctk.CTkFrame(parent)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        setattr(self, f"{name_prefix}_fig", fig)

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        setattr(self, f"{name_prefix}_canvas", canvas)

        # Add toolbar
        toolbar_frame = ctk.CTkFrame(parent)
        toolbar_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Store toolbar reference
        setattr(self, f"{name_prefix}_toolbar", toolbar)

    def _add_initial_guidance(self):
        """Add initial guidance text to empty plots."""
        for tab_name in ["processing", "comparison", "parameters"]:
            fig = getattr(self, f"{tab_name}_fig")
            fig.clf()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Please process data to see visualization",
                    ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            canvas = getattr(self, f"{tab_name}_canvas")
            canvas.draw()

    def update_plots(self, results):
        """Update all plots with new results."""
        if not results:
            return False

        try:
            data, correction_results, trace_time, freq = results

            # Use existing visualization functions with GUI figures
            from flamingo.visualization import plot_data as pd

            # Processing Steps visualization
            pd.visualize_processing_steps(data, trace_time, freq,
                                          fig=self.processing_fig)
            self.processing_canvas.draw()

            # Comparison visualization
            pd.plot_comparison(data, trace_time, freq,
                               fig=self.comparison_fig)
            self.comparison_canvas.draw()

            # Parameters visualization
            pd.plot_correction_parameter(correction_results,
                                         fig=self.parameters_fig)
            self.parameters_canvas.draw()

            return True

        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False