import numpy as np
import scipy.optimize as optimize

from flamingo.utils.config import logger

def fit_periodic_sampling(optimization_dict, trace_time, trace_mean, freq, dt):
    """
    Fit a periodic sampling model to a given time-domain signal using the differential evolution optimization algorithm.

    Parameters:
    optimization_dict (dict): A dictionary containing the optimization parameters.
    trace_time (numpy.ndarray): The time array of the trace.
    trace_mean (numpy.ndarray): The mean of the time-domain signal.
    freq (numpy.ndarray): The frequency array of the trace.
    dt (float): The time step.

    Returns:
    xopt (numpy.ndarray): The optimal parameters for the periodic sampling model (length: 3).
    """
    # Load values from dictionary
    min_values = np.array(list(optimization_dict["min_values"].values()))
    max_values = np.array(list(optimization_dict["max_values"].values()))

    lower_bounds = np.zeros(len(min_values))
    upper_bounds = np.ones(len(min_values))
    freq_limit_idx = np.searchsorted(freq, optimization_dict["frequency_limit"])

    res = optimize.differential_evolution(error_periodic_sampling,
                                          args=(trace_time, trace_mean, dt, freq_limit_idx, min_values, max_values),
                                          maxiter=optimization_dict["max_iteration"],
                                          popsize=optimization_dict["popsize"],
                                          bounds=list(zip(lower_bounds, upper_bounds)))
    xopt = res.x * (max_values - min_values) + min_values

    # Log the fitted periodic parameters in a readable format
    amplitude = xopt[0]
    frequency_hz = xopt[1] / (2 * np.pi)  # Convert rad/s to Hz
    frequency_thz = frequency_hz * 1e-12  # Convert Hz to THz
    phase_rad = xopt[2]

    logger.info("Fitted Periodic Sampling Parameters:")
    logger.info(f"  Amplitude:  {amplitude:.2e}")
    logger.info(f"  Frequency:  {frequency_thz:.1f} THz × 2π")
    logger.info(f"  Phase:      {phase_rad:.1f} rad")

    return xopt


def error_periodic_sampling(x, trace_time, trace_mean, dt, freq_limit_idx, min_values, max_values):
    """
    Calculate the error due to periodic sampling in the given time-domain signal.

    Parameters:
    x (np.ndarray): Fitting parameters, normalized between [0, 1]
    time_trace (np.ndarray): Time array of THz trace (x-axis)
    mean_trace (np.ndarray): Mean of the time-domain signal
    dt (float): Time step
    index_nu (int): Index of the frequency component
    min_values (np.ndarray): Minimum value for periodic sampling
    max_values (np.ndarray): Maximum value for periodic sampling

    Returns:
    error (float): Error due to periodic sampling
    """
    # De-normalize parameter
    x = x * (max_values - min_values) + min_values

    ct = x[0] * np.cos(x[1] * trace_time + x[2])
    corrected = trace_mean - np.gradient(trace_mean, dt) * ct
    error = np.sum(np.abs(np.fft.rfft(corrected)[freq_limit_idx:]))
    return error

def initialize_delay_dilatation(optimization_parameter, trace_fft, freq, trace_time, trace_ref, dt):
    # Load values from dictionary
    min_values = optimization_parameter["min_values"]
    max_values = optimization_parameter["max_values"]

    # Store param_order if provided, otherwise use default
    param_order = optimization_parameter.get("param_order", ["delay", "dilatation", "residual_noise"])

    # Use normalized bounds [0, 1]
    lower_bounds = np.zeros(len(min_values))
    upper_bounds = np.ones(len(min_values))

    # Fit delay and dilatation parameters
    res = optimize.differential_evolution(error_delay_dilatation,
                                          bounds=list(zip(lower_bounds, upper_bounds)),
                                          args=(freq, trace_time, trace_fft, trace_ref, min_values, max_values, dt, param_order),
                                          maxiter=optimization_parameter["max_iteration"],
                                          disp=False)  # Disable progress output

    optimization_parameter["x0"] = res.x
    optimization_parameter["param_order"] = param_order

    return optimization_parameter


def fit_delay_dilatation(optimization_parameter, correction_parameter, trace_fft, freq, trace_time, trace_ref, dt, i):
    """
    Fit a delay and dilatation model to a given frequency-domain signal using the SLSQP optimization algorithm.

    Parameters:
    optimization_parameter (dict): A dictionary containing the optimization parameters.
    correction_parameter (dict): A dictionary containing the correction parameters.
    trace_fft (numpy.ndarray): The FFT of the trace.
    freq (numpy.ndarray): The frequency array of the trace.
    trace_time (numpy.ndarray): The time array of the trace.
    trace_ref (numpy.ndarray): The reference trace.
    dt (float): The time step.
    i (int): The index of the trace.

    Returns:
    optimization_dict (dict): The updated optimization dictionary.
    correction_parameter (dict): The updated correction parameter dictionary.
    """
    # Get parameter order (third return value from get_min_max_arrays)
    min_values = optimization_parameter["min_values"]
    max_values = optimization_parameter["max_values"]
    param_order = optimization_parameter["param_order"]


    x0 = optimization_parameter["x0"]
    # Use normalized bounds [0, 1]
    lower_bounds = np.zeros(len(x0))
    upper_bounds = np.ones(len(x0))
    # Fit delay and dilatation parameters
    res = optimize.minimize(error_delay_dilatation,
                            x0=x0,
                            method="SLSQP",
                            bounds=list(zip(lower_bounds, upper_bounds)),
                            args=(freq, trace_time, trace_fft, trace_ref, min_values, max_values, dt, param_order),
                            options={'maxiter': optimization_parameter["max_iteration"]})
    # Update x0 for next trace with optimal values from this trace, hopefully this leads to faster convergence
    optimization_parameter["x0"] = res.x
    # De-normalize optimal parameters to real values
    x = res.x * (max_values - min_values) + min_values

    for idx, param_name in enumerate(param_order):
        correction_parameter[param_name][i] = x[idx]

    # Always safe to update metrics
    correction_parameter["metrics"]["f_opt"][i] = res.fun
    return optimization_parameter, correction_parameter


def error_delay_dilatation(x, freq, trace_time, trace_fft, trace_ref, min_values, max_values, dt, param_order=None):
    """
    Calculate the error between a corrected trace and a reference trace using the delay and dilatation model.

    Parameters:
    x (numpy.ndarray): The fitting parameters.
    freq (numpy.ndarray): The frequency array of the trace.
    trace_time (numpy.ndarray): The time array of the trace.
    trace_fft (numpy.ndarray): The FFT of the trace.
    trace_ref (numpy.ndarray): The reference trace.
    min_values (numpy.ndarray): The minimum values for the fitting parameters.
    max_values (numpy.ndarray): The maximum values for the fitting parameters.
    dt (float): The time step.

    Returns:
    error (float): The error between the corrected trace and the reference trace.
    """
    # De-normalize optimization parameter
    x = x * (max_values - min_values) + min_values
    trace_corrected = apply_delay_dilatation(x, freq, trace_time, trace_fft, dt, param_order)
    error = np.linalg.norm(trace_ref - trace_corrected) / np.linalg.norm(trace_ref)
    return error


def apply_delay_dilatation(x, freq, trace_time, trace_fft, dt, param_order=None):
    """Apply the effect of delay and dilatation to a given frequency-domain signal."""
    # Default parameter order if not provided
    if param_order is None:
        param_order = ["delay", "dilatation", "residual_noise"]

    # Create parameter dictionary from array using order
    params = {}
    for i, param in enumerate(param_order):
        if i < len(x):
            params[param] = x[i]

    # Apply delay in frequency domain (if parameter exists)
    if "delay" in params:
        z = np.exp(1j * 2 * np.pi * freq * params["delay"])
        trace_delayed = np.fft.irfft(z * trace_fft)
    else:
        trace_delayed = np.fft.irfft(trace_fft)

    # Apply dilatation correction (if parameter exists)
    if "dilatation" in params:
        # Center time at signal peak for better numerical stability
        peak_idx = np.argmax(np.abs(trace_delayed))
        t_center = trace_time[peak_idx]
        t_rel = trace_time - t_center

        # First derivative with optional light smoothing
        gradient_trace = np.gradient(trace_delayed, dt)

        # Apply dilatation correction
        trace_stretched = trace_delayed - params["dilatation"] * t_rel * gradient_trace
    else:
        trace_stretched = trace_delayed

    # Apply amplitude correction (if parameter exists)
    if "residual_noise" in params:
        leftover_noise = 1 - params["residual_noise"]
        trace_corrected = leftover_noise * trace_stretched
    else:
        trace_corrected = trace_stretched

    return trace_corrected