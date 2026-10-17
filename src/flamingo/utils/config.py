from colorama import Fore, Back, Style, just_fix_windows_console
import sys
import logging
from typing import Optional, Dict


class ColoredFormatter(logging.Formatter):
    """Colored log formatter.
    Taken from github-user: joshbode
    Source: https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad
    """

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


formatter = ColoredFormatter(
    '{color}[{levelname}] {message}{reset}',
    style='{', datefmt='%Y-%m-%d %H:%M:%S',
    colors={
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.RED,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
)

# Package logger. Importing flamingo as a library leaves the logging setup of the calling program
# alone: records propagate to the root logger, so warnings and errors still show up by default.
logger = logging.getLogger("flamingo")
logger.setLevel(logging.INFO)


def setup_console_logging():
    """Print colored flamingo log messages to stdout; used when flamingo runs as a program (CLI, GUI)."""
    # Color support for the Windows console, see https://stackoverflow.com/a/61069032
    just_fix_windows_console()
    if not any(getattr(h, "_flamingo_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler._flamingo_console = True
        logger.addHandler(handler)
    # Avoid printing every message twice if the root logger has a handler as well
    logger.propagate = False


def set_debug(DEBUG=False):
    if DEBUG is False:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)


def get_debug():
    if logger.getEffectiveLevel() is logging.WARNING:
        return False
    else:
        return True
