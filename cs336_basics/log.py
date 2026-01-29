import logging
import datetime
from typing import Optional


class UTCISOFormatter(logging.Formatter):
    """Formatter that outputs UTC time in ISO8601 format with 'Z' suffix."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None):
        # record.created is a POSIX timestamp (float)
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        # Example: 2025-12-27T15:50:05Z
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)  # or another level you prefer

    # Avoid affecting other loggers / root logger
    logger.propagate = False

    # Create handler (here: stderr)
    handler: logging.Handler = logging.StreamHandler()

    # Format: logger name, time, PID, TID, level, message
    fmt = (
        "%(asctime)s "  # formatted time (UTC ISO8601 by our formatter)
        "P%(process)d "  # process ID
        "T%(thread)d "  # thread ID
        "[%(levelname)s] "  # log level
        "%(name)s: "  # logger name
        "%(message)s"  # log message
    )

    formatter = UTCISOFormatter(fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    log = get_logger("my_app_logger")
    log.info("Application started")
    log.debug("Debug details here")
