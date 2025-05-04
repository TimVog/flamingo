"""flamingo: Data Correction Tool for THz Time Domain Spectroscopy."""

__version__ = "0.1.0"

# Export key classes/functions for simpler imports
from flamingo.core.processing_pipeline import ProcessingPipeline
from flamingo.main import process_data, main