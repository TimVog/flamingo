"""Command-line interface for Flamingo THz processing tools."""

import argparse

from flamingo.core.optimization_parameter import CorrectionConfig, DEFAULT_SEED
from flamingo.core.processing_pipeline import ProcessingPipeline
from flamingo.utils.config import setup_console_logging
from flamingo.utils.helper_functions import get_trace_count


def parse_arguments():
    """Parse command-line arguments for Flamingo."""
    parser = argparse.ArgumentParser(description="THz-TDS Data Correction Tool")

    # File selection
    parser.add_argument("--input", "-i", required=True, help="Path to input HDF5 file")
    parser.add_argument("--output", "-o", help="Path for saving results (optional)")

    # Processing range
    parser.add_argument("--start", type=int, default=0, help="Starting trace index")
    parser.add_argument("--end", type=int, default=None, help="Ending trace index (default: all traces)")

    # Processing options
    parser.add_argument("--lowcut", type=float, default=0.2e12, help="Low cut frequency (Hz)")
    parser.add_argument("--no-dilatation", action="store_true", help="Disable dilatation correction")
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic correction")
    parser.add_argument("--frequency-limit", type=float, default=None,
                        help="Start of the periodic fit window in Hz (default: auto, noise floor of the mean spectrum)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for the differential evolution fits")

    # Add export option
    parser.add_argument("--export", "-e", help="Path for saving corrected data as HDF5 file")

    return parser.parse_args()


def run_cli():
    """Execute the command-line workflow."""
    args = parse_arguments()
    setup_console_logging()

    filepath = args.input
    if args.end is None:
        args.end = get_trace_count(filepath)

    # Configure correction options
    config = CorrectionConfig()
    config.enabled_corrections["dilatation"] = not args.no_dilatation
    config.enabled_corrections["periodic"] = not args.no_periodic
    config.optimization_settings["seed"] = args.seed
    config.periodic_sampling["frequency_limit"] = args.frequency_limit

    # Process data
    pipeline = ProcessingPipeline(config)
    result = pipeline.process_file(
        filepath, args.start, args.end, args.lowcut
    )

    # Export data if requested
    if args.export:
        pipeline.export_corrected_data(filepath, args.export, args.start, args.end, args.lowcut)

    # Generate visualizations
    from flamingo.visualization import plot_data as pd
    import matplotlib.pyplot as plt

    data, correction_results, trace_time, freq = result
    pd.visualize_processing_steps(data, trace_time, freq)
    pd.plot_comparison(data, trace_time, freq)
    pd.plot_correction_parameter(correction_results)

    plt.show()

    return result