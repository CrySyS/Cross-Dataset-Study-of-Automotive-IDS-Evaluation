import logging
from pathlib import Path
from typing import Optional


_DEF_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, console_level: Optional[str] = None, rich_console=None):
    """
    Configure logging for unified IDS evaluation.
    
    Args:
        level: Logging level for file (INFO, DEBUG, etc.)
        log_file: Optional path to write logs to file (in addition to console)
        console_level: Logging level for console (defaults to WARNING if log_file is set, else matches level)
                      Use "WARNING" to suppress INFO messages in multi-job scenarios
        rich_console: Optional rich.console.Console instance for dashboard-aware output
    
    Returns:
        Logger instance for "unified_ids"
    """
    # Determine console level: if not specified, use WARNING when writing to file (quiet console),
    # otherwise use the main level (verbose console)
    if console_level is None:
        console_level = "WARNING" if log_file is not None else level
    
    # Remove existing handlers to avoid duplicates
    logger = logging.getLogger("unified_ids")
    logger.handlers.clear()
    
    # Set up console handler (less verbose when using file logging)
    if rich_console is not None:
        # Use RichHandler for dashboard-aware output
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                console=rich_console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=False
            )
            console_handler.setLevel(getattr(logging, console_level))
            # RichHandler has its own formatting
        except ImportError:
            # Fallback if rich not available
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, console_level))
            console_handler.setFormatter(logging.Formatter(_DEF_FMT))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level))
        console_handler.setFormatter(logging.Formatter(_DEF_FMT))
    
    # Ensure immediate flushing for console output
    console_handler.flush()
    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, level))  # Logger level should be minimum of all handlers
    
    # Silence matplotlib font debug spam
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    
    # Add file handler if log_file specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(logging.Formatter(_DEF_FMT))
        # Set to flush after each log record for immediate output
        file_handler.addFilter(lambda record: file_handler.stream.flush() or True)
        logger.addHandler(file_handler)
    
    return logger