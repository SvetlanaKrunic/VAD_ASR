# =============================================================================
# utils/logger.py – Python logging configuration
# =============================================================================
# The built-in logging module is used instead of print() because:
#   - It automatically adds timestamps and severity levels (DEBUG, INFO, ERROR...)
#   - It writes to a file AND prints to the terminal simultaneously
#   - Log levels can be adjusted or silenced without changing call sites
#   - It is the standard approach in all production Python projects
# =============================================================================

import logging
import os


def setup_logger(log_path: str = "logs/asr_vad.log") -> logging.Logger:
    """
    Configures and returns a logger instance for the entire project.

    Sets up two handlers:
      1. FileHandler  – writes ALL messages (DEBUG and above) to the log file
      2. StreamHandler – prints only INFO and above to the terminal (less noise)

    Call this function once at program startup and pass the logger to other
    modules. Any module that calls logging.getLogger("asr_vad") will
    automatically receive the same configured logger instance.

    Args:
        log_path (str): Path to the log file. The parent directory is created
                        automatically if it does not exist.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the logs/ directory (and any missing parents) if it does not exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Get or create a logger named "asr_vad"
    # Using a named logger allows all modules to share the same instance by
    # calling logging.getLogger("asr_vad") without repeating the setup
    logger = logging.getLogger("asr_vad")

    # Set the minimum level to DEBUG — individual handlers filter further
    # Level hierarchy: DEBUG < INFO < WARNING < ERROR < CRITICAL
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate log entries if setup_logger is called more than once
    # (e.g., when modules are imported in multiple places)
    if logger.handlers:
        return logger

    # --- File handler: all messages go to the log file ---
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # DEBUG and above → written to file

    # --- Console handler: only INFO and above → printed to terminal ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)    # DEBUG messages are suppressed in the terminal

    # Format: "2024-01-15 14:23:01,234 - INFO - Message here"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Register both handlers on the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger