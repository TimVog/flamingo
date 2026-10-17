"""Utility functions for THz data analysis and configuration."""

from flamingo.utils.config import (
    logger,
    setup_console_logging,
    set_debug,
    get_debug,
    ColoredFormatter
)

from flamingo.utils.helper_functions import (
    create_dict,
    get_filter_coefficients,
    calculate_mean_std,
    normalize_std,
    find_reference_trace
)

from flamingo.utils.log_capture import (
    HybridLogCapture
)