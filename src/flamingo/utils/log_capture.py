"""
Streamlined log capture system for Flamingo GUI.
Handles logger output and redirects stdout/stderr.
Progress bar handling is now done separately by GUIProgressBar.
"""

import sys
import io
import logging

from flamingo.utils.config import logger

class HybridLogCapture:
    """
    A streamlined log capture system that handles logger output in a text widget.
    Progress bar handling is now done separately by GUIProgressBar.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # For log coloring
        self.level_tags = {
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "critical",
            logging.DEBUG: "debug"
        }

    def start_capture(self):
        """Start capturing logger output with robust error handling."""
        # Set up the logger handler
        self.log_handler = logging.Handler()
        self.log_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        self.log_handler.emit = self._log_emit
        logger.addHandler(self.log_handler)

        # Implement custom error handler for logging
        logging.raiseExceptions = False  # Disable error reporting for logging module

        # Create a safer stream redirection approach
        class SafeStreamRedirector:
            def __init__(self, original_stream, text_widget_callback):
                self.original_stream = original_stream
                self.callback = text_widget_callback

            def write(self, string):
                # Only try writing to original stream if it exists and has write method
                if self.original_stream is not None and hasattr(self.original_stream, 'write'):
                    try:
                        self.original_stream.write(string)
                        self.original_stream.flush()
                    except (AttributeError, IOError):
                        pass  # Silently ignore stream errors

                # Process for GUI if not a progress bar output
                if string and not ('%' in string and ('|' in string or '/' in string)) and not '\r' in string:
                    if string.strip():
                        self.callback(string)

            def flush(self):
                if self.original_stream is not None and hasattr(self.original_stream, 'flush'):
                    try:
                        self.original_stream.flush()
                    except (AttributeError, IOError):
                        pass

        # Replace stream redirectors with safe versions
        sys.stdout = SafeStreamRedirector(self.original_stdout,
                                          lambda s: self.text_widget.after(0, self._add_regular_output, s))
        sys.stderr = SafeStreamRedirector(self.original_stderr,
                                          lambda s: self.text_widget.after(0, self._add_regular_output, s))

    def _log_emit(self, record):
        """Custom emit method for logger records."""
        msg = self.log_handler.format(record)
        tag = self.level_tags.get(record.levelno, "default")
        self.text_widget.after(0, self._add_log_text, msg, tag)

    def _add_log_text(self, text, tag):
        """Add logger text to the widget with the appropriate tag."""
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", text + "\n", (tag,))
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")

    def _create_redirector(self, original_stream):
        """Create a redirector for a stream."""
        class StreamRedirector(io.StringIO):
            def write(self_, string):
                # Write to original stream if it exists
                if original_stream is not None:
                    original_stream.write(string)
                    original_stream.flush()

                # Only process if it's not a progress bar output
                if string and not ('%' in string and ('|' in string or '/' in string)) and not '\r' in string:
                    # It's not a progress bar, so display it as regular output
                    if string.strip():
                        self.text_widget.after(0, self._add_regular_output, string)

            def flush(self_):
                if original_stream is not None:
                    original_stream.flush()

        return StreamRedirector()

    def _add_regular_output(self, text):
        """Add regular (non-logger) output to the widget."""
        # Ignore empty strings
        if not text.strip():
            return

        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", text, ("default",))

        # Add newline if text doesn't already end with one
        if not text.endswith('\n'):
            self.text_widget.insert("end", "\n")

        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")

    def stop_capture(self):
        """Stop capturing and restore original settings."""
        # Remove our handler from the logger
        if hasattr(self, 'log_handler'):
            logger.removeHandler(self.log_handler)

        # Restore original stdout and stderr
        if hasattr(sys, 'stdout') and sys.stdout != self.original_stdout:
            sys.stdout = self.original_stdout
        if hasattr(sys, 'stderr') and sys.stderr != self.original_stderr:
            sys.stderr = self.original_stderr