import logging
import datetime
import sys

class ColoredFormatter(logging.Formatter):
    """
    A formatter that adds ANSI color codes to logging output for console display.
    """
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m',     # Reset to default
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        levelname = record.levelname
        # Check if running in a terminal that supports colors
        if sys.stdout.isatty() and levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.name = f"\033[96m{record.name}\033[0m"  # Cyan for the logger name
            record.msg = f"{record.msg}"  # Keep the message uncolored for readability
            # Add timestamp prefix if not already formatted
            if not hasattr(record, 'asctime') or not record.asctime:
                record.asctime = datetime.datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        result = logging.Formatter.format(self, record)
        # Restore the original levelname for other formatters
        record.levelname = levelname
        return result