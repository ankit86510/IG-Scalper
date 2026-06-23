"""
Windows-Safe Logging Utilities
Replaces core/logging_utils.py with Windows-compatible version
"""

import logging
import os
import sys


def fix_windows_console():
    """Fix Windows console encoding for Unicode"""
    if sys.platform.startswith('win'):
        try:
            # Set UTF-8 encoding
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        except AttributeError:
            # Python < 3.7
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')


class WindowsSafeFormatter(logging.Formatter):
    """
    Custom formatter that replaces Unicode symbols with ASCII on Windows
    """

    # Symbol replacements
    SYMBOL_MAP = {
        '✓': '[OK]',
        '✗': '[X]',
        '⚠': '[!]',
        '→': '->',
        '←': '<-',
        '•': '*',
        '★': '*',
        '🤖': '[BOT]',
        '📊': '[DATA]',
        '💰': '[$]',
        '🎯': '[TARGET]',
        '⚡': '[FAST]',
        '🚀': '[GO]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '🛑': '[STOP]',
        '✅': '[DONE]',
        '❌': '[FAIL]',
        '🔍': '[SEARCH]',
        '🔄': '[SYNC]',
        '💡': '[INFO]',
        '⏸': '[PAUSE]',
    }

    def __init__(self, fmt=None, datefmt=None, style='%', use_ascii=None):
        super().__init__(fmt, datefmt, style)
        # Auto-detect if we should use ASCII
        if use_ascii is None:
            use_ascii = sys.platform.startswith('win')
        self.use_ascii = use_ascii

    def format(self, record):
        # Format the message
        result = super().format(record)

        # Replace Unicode symbols on Windows
        if self.use_ascii:
            for unicode_sym, ascii_sym in self.SYMBOL_MAP.items():
                result = result.replace(unicode_sym, ascii_sym)

        return result


def setup_logging(level="INFO", sink="logs/bot.log", use_ascii=None):
    """
    Setup logging with Windows compatibility

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        sink: Log file path
        use_ascii: Force ASCII symbols (None = auto-detect Windows)

    Returns:
        Logger instance
    """
    # Fix Windows console first
    fix_windows_console()

    # Create logs directory
    os.makedirs(os.path.dirname(sink), exist_ok=True)

    # Auto-detect Windows
    if use_ascii is None:
        use_ascii = sys.platform.startswith('win')

    # Create formatters — force Rome timezone in timestamps
    file_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    console_format = '%(asctime)s %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    import pytz
    from datetime import datetime

    class RomeFormatter(WindowsSafeFormatter):
        """Formatter that outputs timestamps in Europe/Rome timezone."""
        _tz_rome = pytz.timezone('Europe/Rome')

        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=self._tz_rome)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime('%Y-%m-%d %H:%M:%S')

    file_formatter = RomeFormatter(file_format, date_format, use_ascii=False)
    console_formatter = RomeFormatter(console_format, date_format, use_ascii=use_ascii)

    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler(sink, encoding='utf-8', errors='replace')
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(file_formatter)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}")
        file_handler = None

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add handlers
    if file_handler:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Return application logger
    logger = logging.getLogger("ig-scalper")

    # Log startup message
    if use_ascii:
        logger.info("=" * 60)
        logger.info("Logging initialized (Windows mode - ASCII symbols)")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("✓ Logging initialized (Unicode mode)")
        logger.info("=" * 60)

    return logger


def safe_log(logger, level, message):
    """
    Safely log a message with error handling

    Args:
        logger: Logger instance
        level: Log level (info, debug, warning, error)
        message: Message to log
    """
    try:
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(str(message))
    except UnicodeEncodeError:
        # Fallback to ASCII
        clean_message = message.encode('ascii', 'replace').decode('ascii')
        log_func(clean_message)
    except Exception as e:
        # Last resort
        print(f"Logging error: {e}")
        print(f"Original message: {message}")


# Convenience functions
def log_success(logger, message):
    """Log success message with checkmark"""
    if sys.platform.startswith('win'):
        safe_log(logger, 'info', f"[OK] {message}")
    else:
        safe_log(logger, 'info', f"✓ {message}")


def log_error(logger, message):
    """Log error message with X mark"""
    if sys.platform.startswith('win'):
        safe_log(logger, 'error', f"[X] {message}")
    else:
        safe_log(logger, 'error', f"✗ {message}")


def log_warning(logger, message):
    """Log warning message with warning symbol"""
    if sys.platform.startswith('win'):
        safe_log(logger, 'warning', f"[!] {message}")
    else:
        safe_log(logger, 'warning', f"⚠ {message}")


def log_info(logger, message):
    """Log info message with info symbol"""
    if sys.platform.startswith('win'):
        safe_log(logger, 'info', f"[i] {message}")
    else:
        safe_log(logger, 'info', f"ℹ {message}")


# Testing
if __name__ == "__main__":
    print("\nTesting Windows-Safe Logging...\n")

    # Setup logger
    logger = setup_logging(level="DEBUG", sink="logs/test_windows.log")

    # Test various log levels
    logger.debug("Debug message - this is a test")
    log_info(logger, "This is an info message")
    log_success(logger, "Operation completed successfully")
    log_warning(logger, "This is a warning message")
    log_error(logger, "This is an error message")

    # Test Unicode symbols
    logger.info("Testing symbols: ✓ ✗ ⚠ → 🤖 📊 💰 🎯")

    # Test with various patterns
    logger.info("✓ Logged in to IG successfully")
    logger.info("📊 Market analysis complete")
    logger.info("🎯 SIGNAL: EUR/USD BUY")
    logger.info("✅ ORDER PLACED: DealRef123")
    logger.info("💰 Daily P&L: +2.5%")
    logger.warning("⚠ Kill switch activated")
    logger.error("✗ Order failed: Insufficient margin")

    print("\n✓ Logging test complete!")
    print("Check logs/test_windows.log for output")