"""Entry point for Flamingo THz data analysis tool."""

import h5py
from flamingo.core.optimization_parameter import CorrectionConfig
from flamingo.core.processing_pipeline import ProcessingPipeline
from flamingo.utils.config import logger


def _get_trace_count(filepath):
    """
    Determine the number of traces in an HDF5 file by counting numeric keys.

    Parameters:
    filepath (str): Path to HDF5 file

    Returns:
    int: Number of traces found in the file
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Count the number of numeric keys in the file (traces)
            trace_keys = [key for key in f.keys() if key.isdigit()]
            return len(trace_keys)
    except Exception as e:
        logger.error(f"Error reading trace count from {filepath}: {e}")
        return 1000  # Default fallback


def process_data(filepath, trace_start=None, trace_end=None, lowcut=0.2e12, config_options=None):
    """
    Process THz data with configurable correction options.

    If trace_start or trace_end are not specified, automatically detects the full range
    of traces available in the file.

    Parameters:
    -----------
    filepath : str
        Path to HDF5 file
    trace_start : int, optional
        Starting trace index. If None, defaults to 0.
    trace_end : int, optional
        Ending trace index. If None, auto-detects from file.
    lowcut : float, optional
        Low frequency cutoff in Hz (default: 0.2e12)
    config_options : dict, optional
        Configuration overrides

    Returns:
    --------
    tuple
        (data, correction_results, trace_time, freq)
    """
    # Auto-detect trace range if not specified
    if trace_start is None:
        trace_start = 0
        logger.info(f"Auto-detected trace_start: {trace_start}")

    if trace_end is None:
        trace_end = _get_trace_count(filepath)
        logger.info(f"Auto-detected trace_end: {trace_end} (total traces available)")

    # Validate the detected/provided range
    if trace_end <= trace_start:
        raise ValueError(f"trace_end ({trace_end}) must be greater than trace_start ({trace_start})")

    # Create configuration with optional overrides
    correction_config = CorrectionConfig()

    if config_options:
        if "enable_dilatation" in config_options:
            correction_config.enabled_corrections["dilatation"] = config_options["enable_dilatation"]
        if "enable_periodic" in config_options:
            correction_config.enabled_corrections["periodic"] = config_options["enable_periodic"]

    # Create processing pipeline
    pipeline = ProcessingPipeline(correction_config)

    # Process data and return results
    return pipeline.process_file(
        filepath, trace_start, trace_end, lowcut
    )


def main():
    """Entry point when executed as a script."""
    from flamingo.cli import run_cli
    return run_cli()


if __name__ == "__main__":
    main()