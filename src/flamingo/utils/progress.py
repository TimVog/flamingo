"""
Custom progress bar implementation for Flamingo.
Provides context-aware progress reporting that works in both CLI and GUI environments.
"""

import os
from tqdm import tqdm

# Global flag to track GUI mode
_gui_mode = False
_gui_progress_bar = None

def set_gui_mode(enabled=True, progress_bar=None):
    """
    Set whether to use GUI progress bars or CLI progress bars.

    Parameters:
    enabled (bool): Whether to enable GUI mode
    progress_bar (GUIProgressBar, optional): The GUI progress bar to use
    """
    global _gui_mode, _gui_progress_bar
    _gui_mode = enabled
    if progress_bar is not None:
        _gui_progress_bar = progress_bar


class GUIProgressBar:
    """A progress bar that updates a tkinter text widget directly."""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.current_bar_id = None
        self.active_bar_position = None

    def update_progress(self, description, current, total, bar_id=None):
        """Update progress in the text widget."""
        # Calculate percentage
        percentage = int(100.0 * current / total)
        bar_length = 20
        filled_length = int(bar_length * current // total)
        # Use # for filled part and - for empty part to match tqdm style
        bar = '#' * filled_length + '-' * (bar_length - filled_length)

        # Format progress string to match tqdm style
        progress_str = f"{description}: {percentage}%|{bar}| {current}/{total}"

        # Update in tkinter's thread
        self.text_widget.after(0, self._update_widget, progress_str, bar_id, percentage == 100)

    def _update_widget(self, progress_str, bar_id, completed):
        """Update the text widget (called in tkinter's thread)."""
        self.text_widget.configure(state="normal")

        # Handle existing bar for this ID
        if bar_id == self.current_bar_id and self.active_bar_position:
            # Delete existing bar line
            line_num = self.active_bar_position.split('.')[0]
            self.text_widget.delete(f"{line_num}.0", f"{line_num}.end")
            position = f"{line_num}.0"
        else:
            # New bar or no position - append to end
            if self.current_bar_id is not None and self.active_bar_position is not None:
                # Add newline before starting a new progress bar
                self.text_widget.insert("end", "\n")
            position = self.text_widget.index("end-1c linestart")

        # Insert the new progress bar
        self.text_widget.insert(position, progress_str, ("progress",))

        # Update tracking
        self.current_bar_id = bar_id if not completed else None
        self.active_bar_position = position if not completed else None

        # Add newline if completed
        if completed:
            self.text_widget.insert("end", "\n")

        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")


class GUITqdm(tqdm):
    """tqdm with GUI progress reporting."""

    def __init__(self, *args, **kwargs):
        # Store the description to use as bar ID
        self.desc = kwargs.get('desc', '')

        # Skip output from tqdm in GUI mode
        if _gui_mode:
            kwargs['file'] = open(os.devnull, 'w')

        # Initialize standard tqdm
        super().__init__(*args, **kwargs)

        # Update GUI immediately if in GUI mode
        if _gui_mode and _gui_progress_bar:
            _gui_progress_bar.update_progress(
                self.desc, 0, self.total, self.desc)

    def update(self, n=1):
        """Override update to also update the GUI in GUI mode."""
        super().update(n)

        if _gui_mode and _gui_progress_bar:
            _gui_progress_bar.update_progress(
                self.desc, self.n, self.total, self.desc)

    def close(self):
        """Override close to finalize the GUI progress bar in GUI mode."""
        if _gui_mode and _gui_progress_bar and not self.disable:
            _gui_progress_bar.update_progress(
                self.desc, self.total, self.total, self.desc)

        super().close()


def smartrange(*args, **kwargs):
    """
    Context-aware progress range that works in both CLI and GUI modes.
    This function automatically uses the right progress bar based on context.
    In CLI mode: Uses standard tqdm
    In GUI mode: Uses GUITqdm with GUI progress bar
    """
    return GUITqdm(range(*args), **kwargs)


# For backward compatibility
guitrange = smartrange