# flamingo

**flamingo** stands for **F**lexible **L**ibrary for **A**mplitude and **M**otion **I**naccuracies **N**eatly **G**one **O**ut :flamingo:  

and is a THz data analysis tool which corrects subtle time distortion errors in Terahertz time-domain spectroscopy (THz-TDS) measurements.

## Origin and Motivation

This project is a fork of [Correct-TDS](https://github.com/THzbiophotonics/Correct-TDS) from the group of [Romain Peretti](https://www.tuscany-erc.fr/). While the original implementation provided valuable algorithms, flamingo reimplements these with several key improvements:

1. **Lightweight design** - Built using minimal dependencies (numpy, scipy, matplotlib) for easier installation and maintenance
2. **Memory efficiency** - Processing one trace at a time, allowing even large datasets (10,000+ traces) to be analyzed on modest hardware
3. **Modern GUI** - Using CustomTkinter instead of PyQT for a lighter-weight graphical user interface
4. **Cross-platform compatibility** - Designed to work seamlessly across operating systems with minimal setup

The scientific foundation remains the excellent work described in:

> E. Denakpo, T. Hannotte, N. Osseiran, F. Orieux and R. Peretti,
> 
> "Signal Estimation and Uncertainties Extraction in Terahertz Time-Domain Spectroscopy,"
> 
> in *IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1-13, 2025, Art no. 6005413,*
> 
> doi: [10.1109/TIM.2025.3554287]()

## What does flamingo do?

flamingo analyzes THz-TDS data and corrects systematic errors that affect measurement precision. It specifically targets data from commercial THz-TDS systems like [Menlo](https://www.menlosystems.com/products/thz-time-domain-solutions/terak15-terahertz-spectrometer/) and [Toptica](https://www.toptica.com/products/terahertz-systems/time-domain), providing tools to reduce the standard deviation across multiple measurements.

The software features:

- **Small footprint** - Minimal dependencies and optimized code for simpler installation and deployment
- **Efficient processing** - Single-trace processing approach that preserves memory even with very large datasets
- **Multiple interfaces** - Python API, command-line tool, and graphical user interface to suit different workflows
- **Standalone executable** - Available as an .exe file for users without Python programming experience

## Overview

flamingo is a comprehensive toolset for analyzing and correcting THz time-domain spectroscopy (THz-TDS) data. It features:

- Efficient processing of raw THz data from HDF-5 (.h5) files
- Memory-optimized algorithms that read one trace at a time, enabling processing of large datasets even on low-spec hardware
- Flexible correction methods for systematic errors in THz-TDS measurements:
  - Delay correction
  - Dilatation correction
  - Periodic error correction ("Ghost" spectrum suppression)
- Interactive visualization tools for both time-domain and frequency-domain data
- Multiple interfaces: Python API, command-line, and graphical user interface

## Installation

### Executable Version

For users without a Python background, a standalone executable version is available:
## Download
- [Latest Release](https://github.com/TimVog/flamingo/releases/latest) - View release notes and all available files
- [Windows Executable](https://github.com/TimVog/flamingo/releases/latest/download/flamingo-v0.1.0-win64.zip) - Direct download for Windows users
 
This allows you to run the application without installing Python or any dependencies.

1. Unpack the `.zip`-file
2. Double-click `flamingo.exe` to launch the application
3. Use the graphical interface to process your THz-TDS data

**Note:** Please scroll down to Usage/Example data to see how you can make your first steps with flamingo.

![GUI Screenshot](images/screenshot_01.png)

### Using pip

```bash
# Install from PyPI
pip install flamingo-thz
```

### From source

```bash
# Clone the repository
git clone https://github.com/TimVog/flamingo.git
cd flamingo

# Install in development mode
pip install -e .
```


## Project Structure

```
flamingo/
├── README.md
├── flamingo.spec      # PyInstaller specification for executable
├── pyproject.toml     # Package configuration
└── src/               # Source code
    └── flamingo/      # Main package
        ├── __init__.py
        ├── cli.py     # Command-line interface
        ├── gui.py     # Graphical user interface
        ├── main.py    # Main API entry points
        ├── algorithms/    # Error correction algorithms
        │   ├── __init__.py
        │   ├── correction_methods.py
        │   └── error_fit_functions.py
        ├── core/          # Core processing functionality
        │   ├── __init__.py
        │   ├── optimization_parameter.py
        │   └── processing_pipeline.py
        ├── gui/           # GUI components
        │   ├── __init__.py
        │   └── components.py
        ├── utils/         # Utility functions
        │   ├── __init__.py
        │   ├── config.py
        │   ├── helper_functions.py
        │   ├── log_capture.py
        │   └── progress.py
        └── visualization/ # Plotting tools
            ├── __init__.py
            └── plot_data.py
```

## Usage

Flamingo provides three different interfaces for flexibility:

### Graphical User Interface

```bash
# Launch the GUI
flamingo-gui
```

![screenshot_02.PNG](images/screenshot_02.png)

![screenshot_03.PNG](images/screenshot_03.png)

The GUI provides an intuitive interface with:
- File selection
- Processing parameter configuration
- Interactive visualizations
- Optimization parameter adjustment
- Export functionality for corrected data

### Python API

```python
from flamingo import process_data

# Process data with custom parameters
data, correction_params, trace_time, freq = process_data(
    filepath='path/to/your/data.h5',
    trace_start=0,
    trace_end=1000,
    lowcut=0.2e12,  # Low cut frequency in Hz
    config_options={
        "enable_dilatation": True,
        "enable_periodic": True
    }
)

# For more advanced usage with full access to the processing pipeline:
from flamingo import ProcessingPipeline
from flamingo.core.optimization_parameter import CorrectionConfig

# Create configuration with custom settings
config = CorrectionConfig()
config.enabled_corrections["dilatation"] = True
config.enabled_corrections["periodic"] = True

# Create processing pipeline
pipeline = ProcessingPipeline(config)

# Process file
result = pipeline.process_file(
    filepath='path/to/your/data.h5',
    trace_start=0,
    trace_end=1000,
    lowcut=0.2e12
)

# Export corrected data to a new file
pipeline.export_corrected_data(
    'path/to/your/data.h5',
    'path/to/output/corrected_data.h5',
    trace_start=0,
    trace_end=1000
)
```

### Command-Line Interface

```bash
# Basic usage
flamingo --input data.h5 --output results/ --start 0 --end 1000 --lowcut 0.2e12

# Disable specific corrections
flamingo --input data.h5 --no-dilatation --no-periodic

# Export corrected data
flamingo --input data.h5 --export corrected_data.h5
```


## Data Format

Flamingo expects H5 files containing THz time-domain spectroscopy data with the following structure:
- Time trace axis stored as `timeaxis`
- Multiple amplitude time traces stored as numeric keys ("0", "1", "2", ...)
- All traces must have the same number of samples and be based on the same time axis

### Example Data

If you'd like to try flamingo but don't have suitable THz-TDS data available, you can download an example file:

1. Get the file `04-12-2023_0mb_50k_100ps.h5` from [this research data repository](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/Y35DYN)
2. This file is already in the correct format for flamingo

**Note**: This example file contains 50 000 traces, which is a large dataset. For an initial exploration of the software's capabilities, try setting the "End trace" parameter to 1 000 in the GUI or via the command line. This is typically sufficient to get a good feel for the software while keeping processing times reasonable.

## Correction Methods

Flamingo implements several correction methods to improve the quality of THz-TDS data:

1. **Delay Correction**: Compensates for time shifts between traces
2. **Dilatation Correction**: Corrects for stretching/compression effects in the time domain
3. **Residual Noise Correction**: Reduces amplitude variations
4. **Periodic Error Correction**: Suppresses "Ghost" spectra caused by periodic sampling errors

## Visualization

The package includes comprehensive visualization tools:
- Time-domain signal plots
- Frequency-domain spectrum plots
- Correction parameter visualization
- Before/after comparison plots

```python
from flamingo.visualization import plot_data

# Generate plots of processed data
plot_data.visualize_processing_steps(data, trace_time, freq)
plot_data.plot_comparison(data, trace_time, freq)
plot_data.plot_correction_parameter(correction_params)
```

## How does this software compare to the other THz-TDS packages like [parrot](https://github.com/puls-lab/parrot) and [phoeniks](https://github.com/puls-lab/phoeniks)?

- **parrot** :parrot: is a fully-fledged solution for continuously recorded THz data with two raw streams (position and THz signal). It takes care of cutting, correcting the phase shift between these two streams, interpolation and so on. The presented algorithms are also going into parrot to keep it as a complete package.
- **phoeniks** :bird: is at the end of the processing chain, if one is interested in the complex refractive index. It uses already averaged data as an input and needs a sample and reference trace.
- **flamingo** :flamingo: is more specialized, using already interpolated traces like they can be recorded with commercial THz-TDS. By correcting subtle errors, which can be detected having an ensemble of traces, the standard deviation of the dataset can be reduced, improving phase accuracy, which is a pre-requisite for accurate extraction of the complex refractive index.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

- Original [Correct-TDS](https://github.com/THzbiophotonics/Correct-TDS) implementation by [Romain Peretti's group](https://www.tuscany-erc.fr/)
- Algorithms based on the paper by E. Denakpo et al. (2025)
- Package development by Tim Vogel
- GUI development and object-oriented architecture support by Claude 3.7 Sonnet, which provided assistance in restructuring the codebase from linear scripts to a modular, object-oriented package with a modern graphical interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.
