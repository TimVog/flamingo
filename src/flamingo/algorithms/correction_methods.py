import numpy as np

def apply_delay_correction(trace_fft, freq, delay):
    """
    Apply delay correction in frequency domain.

    Parameters:
    trace_fft (numpy.ndarray): The FFT of the trace
    freq (numpy.ndarray): Frequency array
    delay (float): Delay parameter in seconds

    Returns:
    numpy.ndarray: Time-domain signal with delay correction applied
    """
    z = np.exp(1j * 2 * np.pi * freq * delay)
    return np.fft.irfft(z * trace_fft)

def apply_dilatation_correction(trace, trace_time, dt, dilatation):
    """
    Apply dilatation correction using the gradient approach.

    Parameters:
    trace (numpy.ndarray): Time-domain signal
    trace_time (numpy.ndarray): Time array
    dt (float): Time step
    dilatation (float): Dilatation parameter

    Returns:
    numpy.ndarray: Time-domain signal with dilatation correction applied
    """
    # Center time at signal peak for better stability
    peak_idx = np.argmax(np.abs(trace))
    t_center = trace_time[peak_idx]
    t_rel = trace_time - t_center

    # Calculate gradient
    grad_trace = np.gradient(trace, dt)

    # Apply correction
    return trace - dilatation * t_rel * grad_trace

def apply_amplitude_correction(trace, residual_noise):
    """
    Apply amplitude scaling.

    Parameters:
    trace (numpy.ndarray): Time-domain signal
    residual_noise (float): Residual noise parameter

    Returns:
    numpy.ndarray: Amplitude-corrected signal
    """
    leftover_noise = 1 - residual_noise
    return leftover_noise * trace

def apply_periodic_correction(trace, trace_time, dt, periodic_params):
    """
    Apply periodic sampling error correction.

    Parameters:
    trace (numpy.ndarray): Time-domain signal
    trace_time (numpy.ndarray): Time array
    dt (float): Time step
    periodic_params (numpy.ndarray): Parameters [amplitude, frequency, phase]

    Returns:
    numpy.ndarray: Signal with periodic correction applied
    """
    # Calculate cosine modulation term
    ct = periodic_params[0] * np.cos(periodic_params[1] * trace_time + periodic_params[2])

    # Apply correction
    return trace - np.gradient(trace, dt) * ct

def apply_delay_dilatation_correction(trace_fft, trace_time, freq, dt, params, enabled_corrections):
    """
    Apply all enabled corrections (except periodic) in the proper sequence.

    Parameters:
    trace_fft (numpy.ndarray): FFT of time-domain signal
    trace_time (numpy.ndarray): Time array
    freq (numpy.ndarray): Frequency array
    dt (float): Time step
    params (dict): Dictionary of correction parameters
    enabled_corrections (dict): Dictionary of boolean flags for each correction

    Returns:
    numpy.ndarray: Fully corrected time-domain signal
    """
    # Use dictionary access instead of positional access
    if enabled_corrections["delay"] and "delay" in params:
        trace_corrected = apply_delay_correction(trace_fft, freq, params["delay"])
    else:
        trace_corrected = np.fft.irfft(trace_fft)

    if enabled_corrections["dilatation"] and "dilatation" in params:
        trace_corrected = apply_dilatation_correction(
            trace_corrected, trace_time, dt, params["dilatation"])

    if "residual_noise" in params:
        trace_corrected = apply_amplitude_correction(
            trace_corrected, params["residual_noise"])

    return trace_corrected