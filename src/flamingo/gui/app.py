"""
Flamingo - THz-TDS Data Analysis GUI
This module provides a graphical user interface for the Flamingo THz-TDS data processing package.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Import Flamingo package components
from flamingo.core.optimization_parameter import CorrectionConfig
from flamingo.core.processing_pipeline import ProcessingPipeline
from flamingo.utils.log_capture import HybridLogCapture
from flamingo.utils.config import logger

# Import GUI components
from flamingo.gui.components import ControlPanel, VisualizationPanel, OptimizationPanel

# Set appearance mode and default theme
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"


class FlamingoGUI(ctk.CTk):
    """Main GUI class for Flamingo THz-TDS Data Analysis Tool."""

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Flamingo - THz-TDS Data Analysis")
        self.geometry("1200x800")
        self.minsize(900, 700)

        # Initialize processing state variables
        self.processing_complete = False
        self.processing_in_progress = False
        self.results = None

        # Create GUI layout
        self.create_layout()

        # Setup console capture
        self.setup_console_capture()

        # Initialize GUI progress bar system and enable GUI mode
        from flamingo.utils.progress import GUIProgressBar, set_gui_mode
        self.progress_bar = GUIProgressBar(self.log_text)
        set_gui_mode(True, self.progress_bar)

    def setup_console_capture(self):
        """Set up hybrid logger and console output capture for the log window."""
        # Create and start the hybrid capture system
        self.log_capture = HybridLogCapture(self.log_text)
        self.log_capture.start_capture()

        # Log initial message
        logger.info("Flamingo GUI started - Log output redirected to this window")
        logger.info("Console capture initialized")

    def create_layout(self):
        """Create the main GUI layout with log window at bottom."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=8)  # Main content area (80%)
        self.grid_rowconfigure(1, weight=2)   # Log area (20%)

        # Initialize control panel component
        self.control_panel = ControlPanel(
            self,
            callbacks={
                'browse': self.browse_file_callback,
                'process': self.process_data_callback,
                'export': self.export_data_callback
            }
        )
        self.control_panel.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Initialize visualization panel component
        self.visualization_panel = VisualizationPanel(self)
        self.visualization_panel.frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Add log window
        self.setup_log_window()

    def update_visualizations(self):
        """Update all visualization tabs with processed data."""
        if not self.results:
            return

        success = self.visualization_panel.update_plots(self.results)
        if success:
            logger.info("Visualizations updated successfully")
        else:
            logger.error("Error updating visualizations")

    def show_error_message(self, error_message):
        """Show error message and reset UI."""
        self.processing_in_progress = False
        self.control_panel.set_status("Processing failed")
        messagebox.showerror("Processing Error", f"An error occurred: {error_message}")
        # Log the error too
        logger.error(f"Processing error: {error_message}")

    def setup_log_window(self):
        """Set up the logging window at the bottom of the GUI."""
        # Create frame for log window
        self.log_frame = ctk.CTkFrame(self)
        self.log_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(0, weight=0)  # Label
        self.log_frame.grid_rowconfigure(1, weight=1)  # Text widget

        # Add a label
        log_label = ctk.CTkLabel(
            self.log_frame,
            text="Log Output",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        log_label.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        # Create a frame for the text widget
        text_frame = ctk.CTkFrame(self.log_frame)
        text_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        # Dark theme colors
        bg_color = "#1e1e1e"  # Dark gray for background
        text_color = "#f0f0f0"  # Light gray for default text

        # Create the text widget with dark theme
        self.log_text = tk.Text(
            text_frame,
            height=12,
            width=50,
            wrap="none",
            font=("Courier", 12),
            bg=bg_color,
            fg=text_color,
            borderwidth=0,
            padx=5,
            pady=5
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Add scrollbar to text widget
        scrollbar = ctk.CTkScrollbar(text_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Configure text tags for different log levels and progress bars
        self.log_text.tag_configure("info", foreground="#2ca02c")       # tab:green
        self.log_text.tag_configure("warning", foreground="#ff7f0e")    # tab:orange
        self.log_text.tag_configure("error", foreground="#d62728")      # tab:red
        self.log_text.tag_configure("critical", foreground="#e377c2")   # tab:pink
        self.log_text.tag_configure("debug", foreground="#AAAAAA")      # Gray
        self.log_text.tag_configure("progress", foreground="#55AAFF")   # tab:blue
        self.log_text.tag_configure("default", foreground="#1f77b4")    # Default text color

        # Make text read-only initially
        self.log_text.configure(state="disabled")

    def browse_file_callback(self, control_panel):
        """Open file dialog to select input HDF5 file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if file_path:
            control_panel.set_file_path(file_path)
            logger.info(f"Selected file: {os.path.basename(file_path)}")
            self.update_trace_range_from_file(file_path, control_panel)

    def update_trace_range_from_file(self, file_path, control_panel):
        """Update trace range based on selected file content."""
        try:
            import h5py
            with h5py.File(file_path, "r") as f:
                # Count the number of numeric keys in the file (traces)
                trace_keys = [key for key in f.keys() if key.isdigit()]
                if trace_keys:
                    max_trace = len(trace_keys)
                    control_panel.set_trace_range(0, max_trace)
                    logger.info(f"Detected {max_trace} traces in file, updated 'End Trace' accordingly.")
                else:
                    # Default if no traces found
                    control_panel.set_trace_range(0, 1000)
                    logger.warning("No traces detected in file, using default range")
        except Exception as e:
            logger.error(f"Error reading trace count: {e}")
            control_panel.set_trace_range(0, 1000)

    def process_data_callback(self, control_panel):
        """Process the data with the selected settings."""
        # Get parameters from control panel
        params = control_panel.get_parameters()

        # Validate inputs
        if not os.path.isfile(params['file_path']) or not params['file_path'].endswith('.h5'):
            messagebox.showerror("Error", "Please select a valid HDF5 file")
            logger.error("No valid HDF5 file selected")
            return

        if params['trace_end'] <= params['trace_start']:
            messagebox.showerror("Error", "End trace must be greater than start trace")
            logger.error("Invalid trace range - end must be greater than start")
            return

        if params['lowcut'] <= 0:
            messagebox.showerror("Error", "Low cut frequency must be positive")
            logger.error("Invalid low cut frequency")
            return

        if self.processing_in_progress:
            messagebox.showinfo("Processing", "Data processing is already in progress")
            logger.warning("Processing already in progress")
            return

        # Get optimization parameters from the optimization panel
        optimization_params = self.visualization_panel.optimization_panel.get_optimization_parameters()
        if optimization_params is None:
            messagebox.showerror("Error", "Invalid optimization parameters")
            logger.error("Invalid optimization parameters")
            return

        # Add optimization parameters to the processing parameters
        params['optimization_params'] = optimization_params

        # Start processing in a separate thread
        self.processing_in_progress = True
        control_panel.set_status("Processing data...")
        logger.info(f"Starting data processing for traces {params['trace_start']} to {params['trace_end']}")
        logger.info(f"Dilatation correction: {params['enable_dilatation']}, Periodic correction: {params['enable_periodic']}")

        # Start processing thread
        processing_thread = threading.Thread(target=self.run_processing, args=(params,))
        processing_thread.daemon = True
        processing_thread.start()

    def run_processing(self, params):
        """Run the data processing in a separate thread."""
        try:
            # Create configuration with user settings
            correction_config = CorrectionConfig()
            correction_config.enabled_corrections["dilatation"] = params['enable_dilatation']
            correction_config.enabled_corrections["periodic"] = params['enable_periodic']

            # Update correction configuration with user-defined optimization parameters
            if 'optimization_params' in params:
                opt_params = params['optimization_params']

                # Update bounds
                if 'bounds' in opt_params:
                    correction_config.bounds = opt_params['bounds']

                # Update periodic sampling parameters
                if 'periodic_sampling' in opt_params:
                    correction_config.periodic_sampling = opt_params['periodic_sampling']

                # Update optimization settings
                if 'optimization_settings' in opt_params:
                    correction_config.optimization_settings = opt_params['optimization_settings']

                logger.info("Applied custom optimization parameters")

            # Create processing pipeline
            pipeline = ProcessingPipeline(correction_config)

            # Process data
            self.results = pipeline.process_file(
                params['file_path'],
                params['trace_start'],
                params['trace_end'],
                params['lowcut']
            )

            # Update UI after processing is complete
            self.after(100, self.processing_complete_callback)

        except Exception as e:
            # Store the error and handle it in the main thread
            error_message = str(e)
            self.after(100, lambda error=error_message: self.show_error_message(error))
            logger.error(f"Processing error: {error_message}")

    def processing_complete_callback(self):
        """Called when processing is complete to update UI."""
        self.processing_in_progress = False
        self.processing_complete = True
        self.control_panel.set_status("Processing complete")
        self.control_panel.set_export_enabled(True)
        logger.info("Processing completed successfully")

        # Update visualizations
        self.update_visualizations()

    def export_data_callback(self, control_panel):
        """Export corrected data to a new HDF5 file."""
        if not self.processing_complete:
            messagebox.showinfo("Export", "Please process data first")
            logger.warning("Export attempted without processed data")
            return

        params = control_panel.get_parameters()

        # Get export file path
        export_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            initialdir=os.path.dirname(params['file_path']),
            initialfile=f"corrected_{os.path.basename(params['file_path'])}"
        )

        if not export_path:
            return

        try:
            # Create configuration with user settings
            correction_config = CorrectionConfig()
            correction_config.enabled_corrections["dilatation"] = params['enable_dilatation']
            correction_config.enabled_corrections["periodic"] = params['enable_periodic']

            # Create processing pipeline
            pipeline = ProcessingPipeline(correction_config)

            # Set up progress tracking
            control_panel.set_status("Exporting data...")
            logger.info(f"Exporting corrected data to {os.path.basename(export_path)}")

            # Export data
            pipeline.export_corrected_data(
                params['file_path'],
                export_path,
                params['trace_start'],
                params['trace_end']
            )

            # Update UI
            control_panel.set_status("Export complete")
            logger.info("Export completed successfully")
            messagebox.showinfo("Export", f"Data exported successfully to {export_path}")

        except Exception as e:
            control_panel.set_status("Export failed")
            logger.error(f"Export error: {str(e)}")
            messagebox.showerror("Export Error", f"An error occurred: {str(e)}")

    def __del__(self):
        """Clean up resources when GUI is closed."""
        # Stop console capture
        if hasattr(self, 'log_capture'):
            self.log_capture.stop_capture()


def main():
    """Main entry point for the Flamingo GUI application."""
    app = FlamingoGUI()
    app.mainloop()


if __name__ == "__main__":
    main()