import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import warnings


def visualize_processing_steps(data, trace_time, freq, fig=None):
    """
    Creates visualization of processing steps.

    Parameters
    ----------
    data : dict
        Nested dictionary containing processed data
    trace_time : ndarray
        Time axis values
    freq : ndarray
        Frequency array in Hz
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates new figure.

    Returns
    -------
    tuple
        (fig, axs) - Figure and axes objects
    """
    # Filter processing steps to only include those with non-zero data
    # This ensures disabled corrections don't show empty panels
    valid_steps = []
    for step in data.keys():
        if np.any(data[step]["time"]["mean"] != 0) or np.any(data[step]["freq"]["mean"] != 0):
            valid_steps.append(step)

    # Handle case where no valid steps are found
    if not valid_steps:
        valid_steps = ["raw"]  # Always include raw data at minimum

    # Suppress warnings before figure operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create figure if not provided
        if fig is None:
            fig, axs = plt.subplots(
                nrows=2, ncols=len(valid_steps),
                figsize=(4*len(valid_steps), 8),
                sharex="row", sharey="row",
                constrained_layout=True
            )
        else:
            # Clear existing figure and create proper subplot layout
            fig.clear()
            axs = fig.subplots(2, len(valid_steps), sharex="row", sharey="row")

    # Handle case with only one column
    if len(valid_steps) == 1:
        axs = np.array(axs).reshape(2, 1)

    # Plot each processing step
    for j, process_step in enumerate(valid_steps):
        # Get display names based on actual processing steps
        display_name = process_step

        # Always use original names - these are already correct in the data dictionary
        axs[0, j].set_title(display_name)

        # Time domain plots (top row)
        time_ps = trace_time * 1e12  # Convert to picoseconds
        axs[0, j].plot(time_ps, data[process_step]["time"]["mean"],
                       label="Mean", color='tab:blue')
        axs[0, j].plot(time_ps, data[process_step]["time"]["std"],
                       label="Std Dev", color='tab:red', alpha=0.9)
        axs[0, j].set_xlabel("Time (ps)")
        axs[0, j].set_ylabel("Amplitude (V)")
        axs[0, j].grid(True, alpha=0.3)

        # Frequency domain plots (bottom row)
        freq_thz = freq * 1e-12  # Convert to THz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Handle zero values for log plotting to avoid warnings
            mean_data = np.abs(data[process_step]["freq"]["mean"])
            std_data = np.abs(data[process_step]["freq"]["std"])

            # Replace zeros with minimum non-zero value to avoid log scale issues
            if np.any(mean_data):
                min_nonzero = np.min(mean_data[mean_data > 0]) if np.any(mean_data > 0) else 1e-10
                mean_data[mean_data <= 0] = min_nonzero * 0.1

            if np.any(std_data):
                min_nonzero = np.min(std_data[std_data > 0]) if np.any(std_data > 0) else 1e-10
                std_data[std_data <= 0] = min_nonzero * 0.1

            axs[1, j].semilogy(freq_thz, mean_data, label="Mean", color='tab:blue')
            axs[1, j].semilogy(freq_thz, std_data, label="Std Dev", color='tab:red', alpha=0.9)

        axs[1, j].set_xlabel("Frequency (THz)")
        axs[1, j].set_ylabel("Amplitude")
        axs[1, j].grid(True, alpha=0.3)

    # Add legend to first column only (applies to all)
    axs[0, 0].legend()
    axs[1, 0].legend()

    return fig, axs

def plot_correction_parameter(correction_parameters, fig=None):
    """
    Visualize correction parameters and optimization metrics across traces.

    Parameters
    ----------
    correction_parameters : dict
        Dictionary containing fitted correction parameters and metrics
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates new figure.

    Returns
    -------
    tuple
        (fig, axs) - Figure and axes objects for visualization
    """
    # Extract plottable parameters (exclude periodic_params and dictionaries)
    plottable_params = [param for param, value in correction_parameters.items()
                        if isinstance(value, np.ndarray) and value.ndim == 1
                        and len(value) > 0  # Make sure array is not empty
                        and param != "periodic_params"]

    # Add f_opt from metrics if available and not empty
    if ("metrics" in correction_parameters and
            "f_opt" in correction_parameters["metrics"] and
            len(correction_parameters["metrics"]["f_opt"]) > 0):
        plottable_params.append("f_opt (optimization error)")

    # Handle case with no plottable parameters
    if not plottable_params:
        if fig is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.clear()
                ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No parameters to display with current settings",
                    ha="center", va="center")
            ax.set_axis_off()
            return fig, [ax]
        else:
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, "No parameters to display with current settings",
                    ha="center", va="center")
            ax.set_axis_off()
            return fig, [ax]

    # Create or clear figure
    if fig is None:
        fig, axs = plt.subplots(
            nrows=len(plottable_params), ncols=1,
            figsize=(10, 2 * len(plottable_params)),
            sharex=True,
            constrained_layout=True
        )
    else:
        # Clear existing figure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.clear()
            axs = fig.subplots(len(plottable_params), 1, sharex=True)

    # Handle single parameter case
    if len(plottable_params) == 1:
        axs = [axs]

    # Plot each parameter
    for i, param_name in enumerate(plottable_params):
        if param_name == "f_opt (optimization error)":
            # Plot f_opt with distinctive formatting
            axs[i].plot(correction_parameters["metrics"]["f_opt"],
                        color='tab:red')
            axs[i].set_ylabel("Error Value")
            axs[i].set_title("Optimization Error")
        else:
            # Plot regular parameters
            axs[i].plot(correction_parameters[param_name])
            axs[i].set_title(param_name)

            # Apply special formatting for delay parameter
            if param_name == "delay":
                axs[i].yaxis.set_major_formatter(EngFormatter("s"))

        axs[i].grid(True, alpha=0.3)

    # Add common x-axis label
    axs[-1].set_xlabel("Trace Number")

    return fig, axs

def plot_comparison(data, trace_time, freq, fig=None):
    """
    Compare processing steps with overlaid mean values and STD bands.
    Uses engineering notation for units and matplotlib's default color cycle.

    Parameters:
    ----------
    data : dict
        Nested dictionary with processing steps, domains, and mean/std values
    trace_time : ndarray
        Time array in seconds
    freq : ndarray
        Frequency array in Hz
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates new figure.

    Returns
    -------
    tuple
        (fig, axs) - Figure and axes objects
    """
    # Suppress warnings for figure creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create or clear figure
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        else:
            # Clear existing figure completely
            fig.clear()
            ax1, ax2 = fig.subplots(2, 1)

    alpha_fill = 0.2

    # Set engineering formatters for axes
    ax1.xaxis.set_major_formatter(EngFormatter("s"))
    ax1.yaxis.set_major_formatter(EngFormatter("V"))

    # Filter for only valid steps with non-zero data
    valid_steps = []
    for step in data.keys():
        if np.any(data[step]["time"]["mean"] != 0) or np.any(data[step]["freq"]["mean"] != 0):
            valid_steps.append(step)

    # Plot time domain with automatic color cycle - only for valid steps
    for step in valid_steps:
        # Use original labels - these are already correctly generated by the pipeline
        display_name = step

        mean = data[step]['time']['mean']
        std = data[step]['time']['std']

        line, = ax1.plot(trace_time, mean, label=display_name, linewidth=1.5)
        ax1.fill_between(trace_time, mean-std, mean+std,
                         color=line.get_color(), alpha=alpha_fill)

    ax1.set_title('Time Domain Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Plot frequency domain (log scale) - only for valid steps
    freq_thz = freq * 1e-12  # Convert to THz for readability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in valid_steps:
            # Use original names - consistent with above
            display_name = step

            mean = np.abs(data[step]['freq']['mean'])
            # Avoid plotting zero values on log scale
            if np.any(mean):
                min_nonzero = np.min(mean[mean > 0]) if np.any(mean > 0) else 1e-10
                mean[mean <= 0] = min_nonzero * 0.1

            line, = ax2.semilogy(freq_thz, mean, label=display_name, linewidth=1.5)

    ax2.set_title('Frequency Domain Comparison')
    ax2.set_xlabel('Frequency (THz)')  # Keep THz for frequency domain
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    return fig, (ax1, ax2)