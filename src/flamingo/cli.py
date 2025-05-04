"""Command-line interface for Flamingo THz processing tools."""

import argparse

from flamingo.core.optimization_parameter import CorrectionConfig
from flamingo.core.processing_pipeline import ProcessingPipeline


def parse_arguments():
    """Parse command-line arguments for Flamingo."""
    parser = argparse.ArgumentParser(description="THz-TDS Data Correction Tool")

    # File selection
    parser.add_argument("--input", "-i", help="Path to input HDF5 file")
    parser.add_argument("--output", "-o", help="Path for saving results (optional)")

    # Processing range
    parser.add_argument("--start", type=int, default=0, help="Starting trace index")
    parser.add_argument("--end", type=int, default=1000, help="Ending trace index")

    # Processing options
    parser.add_argument("--lowcut", type=float, default=0.2e12, help="Low cut frequency (Hz)")
    parser.add_argument("--no-dilatation", action="store_true", help="Disable dilatation correction")
    parser.add_argument("--no-periodic", action="store_true", help="Disable periodic correction")

    # Add export option
    parser.add_argument("--export", "-e", help="Path for saving corrected data as HDF5 file")

    return parser.parse_args()


def run_cli():
    """Execute the command-line workflow."""
    args = parse_arguments()

    # Use provided file or default
    filepath = args.input if args.input else r"C:\Users\Tim\Downloads\04-12-2023_0mb_50k_100ps.h5"

    # Configure correction options
    config = CorrectionConfig()
    config.enabled_corrections["dilatation"] = not args.no_dilatation
    config.enabled_corrections["periodic"] = not args.no_periodic

    # Process data
    pipeline = ProcessingPipeline(config)
    result = pipeline.process_file(
        filepath, args.start, args.end, args.lowcut
    )

    # Export data if requested
    if args.export:
        pipeline.export_corrected_data(filepath, args.export, args.start, args.end)

    # Generate visualizations
    from flamingo.visualization import plot_data as pd
    import matplotlib.pyplot as plt

    data, correction_results, trace_time, freq = result
    pd.visualize_processing_steps(data, trace_time, freq)
    pd.plot_comparison(data, trace_time, freq)
    pd.plot_correction_parameter(correction_results)

    plt.show()

    return result