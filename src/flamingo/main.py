"""Entry point for Flamingo THz data analysis tool."""

from flamingo.core.optimization_parameter import CorrectionConfig
from flamingo.core.processing_pipeline import ProcessingPipeline


def process_data(filepath, trace_start, trace_end, lowcut, config_options=None):
    """
    Process THz data with configurable correction options.

    Parameters:
    filepath (str): Path to HDF5 file
    trace_start (int): Starting trace index
    trace_end (int): Ending trace index
    lowcut (float): Low frequency cutoff in Hz
    config_options (dict, optional): Configuration overrides

    Returns:
    tuple: (data, correction_results, trace_time, freq)
    """
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