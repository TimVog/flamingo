import numpy as np
from scipy.signal import butter
from matplotlib.ticker import EngFormatter


def create_dict(trace_time):
    """
    Create a nested dictionary to store the mean and standard deviation.

    The nested structure is as follows:
    We have three processing steps ("raw", "filtered_windowed", "corrected (delay_dilatation)", "corrected(+periodic)")
    and two domains ("time", "freq"). For each processing step and domain, the function creates a dictionary
    with two keys ("mean" and "std"), and initializes the values to zero arrays of the appropriate length.
    We use rfft to get the correct length of samples for the frequency domain
    (we are only interested in the correct length, not in its content).

    Parameters:
    trace_time (numpy.ndarray): The time array (x-axis) of the traces.

    Returns:
    data (dict): The nested dictionary initialized with numpy-arrays (zeros).
    """
    trace_time_fft = np.fft.rfft(trace_time)
    data = {}
    for processing_step in ["raw", "filtered_windowed", "corrected (delay_dilatation)", "corrected (+periodic)"]:
        data[processing_step] = {}
        for domain in ["time", "freq"]:
            data[processing_step][domain] = {
                "mean": np.zeros(len(trace_time) if domain == "time" else len(trace_time_fft)),
                "std": np.zeros(len(trace_time) if domain == "time" else len(trace_time_fft))}
    return data


def get_filter_coefficients(fs, lowcut=None, highcut=None, order=None):
    """
    Calculate the filter coefficients for a given sampling frequency with specific lowcut- and highcut-frequencies.

    Parameters:
    fs (float): The sampling frequency in (s).
    lowcut (float, optional): The low-cut frequency for the high-pass filter in (Hz).
    highcut (float, optional): The high-cut frequency for the low-pass filter in (Hz).
    order (int, optional): The order of the filter.

    Returns:
    sos (numpy.ndarray): The filter coefficients in second-order sections format.
    """
    nyquist_frequency = 0.5 * fs
    if highcut is not None and highcut > nyquist_frequency:
        print(
            f"{EngFormatter('Hz')(highcut)} > Nyquist-frequency ({EngFormatter('Hz')(nyquist_frequency)}), ignoring parameter.")
        highcut = None
    if lowcut is not None and highcut is not None:
        low = lowcut / nyquist_frequency
        high = highcut / nyquist_frequency
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    elif highcut is not None:
        low = highcut / nyquist_frequency
        sos = butter(order, low, analog=False, btype='low', output='sos')
    elif lowcut is not None:
        high = lowcut / nyquist_frequency
        sos = butter(order, high, analog=False, btype='high', output='sos')
    else:
        raise NotImplementedError("Lowcut and highcut need to be specified either with a frequency or 'None'.")
    return sos


def calculate_mean_std(trace, count, process_step, domain, data):
    """
    Calculate the mean and standard deviation of a trace using Welford's algorithm. This makes it possible to
    calculate mean and standard deviation in one loop through a dataset without loading the complete dataset at once.
    For more information see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Parameters:
    trace (numpy.ndarray): A single trace (currently read-into memory from file).
    count (int): The current trace index (starting at 0).
    process_step (str): The processing step ("raw", "filtered_windowed" or "corrected").
    domain (str): The domain ("time" or "freq").
    data (dict): The current data dictionary.

    Returns:
    data (dict): The updated data dictionary containing mean and std.
    """
    delta = trace - data[process_step][domain]["mean"]
    data[process_step][domain]["mean"] = data[process_step][domain]["mean"] + delta / (count + 1)
    data[process_step][domain]["std"] = data[process_step][domain]["std"] + delta * (
            trace - data[process_step][domain]["mean"])
    return data


def normalize_std(number_of_traces, data, process_step=None, domain=None):
    """
    Normalize the standard deviation of the data in the dictionary, which is necessary when using Welford's algorithm.

    Parameters:
    number_of_traces (int): The number of traces in the dataset.
    data (dict): The data dictionary containing mean and standard deviation.
    process_step (list, optional): The processing step(s) to normalize. If None, all processing steps are normalized.
    domain (list, optional): The domain(s) to normalize. If None, both time and frequency domains are normalized.

    Returns:
    data (dict): The updated data dictionary with normalized standard deviation.
    """
    if process_step is None:
        process_step = data.keys()
    if domain is None:
        domain_list = ["time", "freq"]
    for key in process_step:
        for domain in domain_list:
            data[key][domain]["std"] = np.sqrt(data[key][domain]["std"] / number_of_traces)
    return data


def find_reference_trace(trace_mean_normed, trace_windowed):
    """
    Find a single trace based on cosine similarity which has the closest distance to the mean trace.
    The advantage of the cosine similarity, compared to other norms (e.g., Euclidian L2 norm) is,
    that it is insensitive to the magnitude of the vector/amplitude of the data.

    Parameters:
    trace_mean_normed (np.array): Normalized mean (windowed) trace in time domain
    trace_windowed (np.array): Current (windowed) trace in time domain

    Returns:
    distance (float): Calculated cosine similarity (distance) between current trace and mean trace
    """
    distance = 1 - np.dot(trace_windowed, trace_mean_normed)
    return distance
