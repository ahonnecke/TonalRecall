# Modular Logger with Per-Module Log Levels

This document describes a Python logging system that supports **per-module log level configuration**, allowing developers to increase or decrease the verbosity of specific components independently.

## Purpose

In complex applications, different parts of the system may require different levels of logging detail. For example:

* A noisy audio analysis module might be set to `WARNING` to reduce clutter.
* A debugging phase in a note matching algorithm might need `DEBUG` level verbosity.

A modular logger makes this easy to manage.

---

## Features

* Centralized logging configuration.
* Explicit log level control per module.
* Prevents log message duplication with `propagate = False`.
* Avoids handler duplication by caching loggers and using a shared handler.

---

## How It Works

1. **Define a `MODULE_LOG_LEVELS` mapping**:

   ```python
   MODULE_LOG_LEVELS = {
       "tonal": logging.INFO,
       "tonal.audio": logging.WARNING,
       "tonal.matcher": logging.DEBUG,
   }
   ```

2. **Initialize logging once with `setup_logging()`**:

   * Configures each module logger individually.
   * Attaches a shared console handler with a consistent format.

3. **Use `get_logger(module_name)` to get a configured logger**:

   * Only allows modules listed in `MODULE_LOG_LEVELS`.
   * Prevents unintended logging behavior.

---

## Usage Example

```python
from logger_config import setup_logging, get_logger

setup_logging()

logger = get_logger("tonal.matcher")
logger.debug("This message will be shown if matcher is set to DEBUG")
```

---

## Benefits

* Encourages intentional logging by requiring explicit configuration.
* Simplifies debugging specific modules without overwhelming output.
* Prevents accidental log flooding from third-party libraries.

---

## Suggested Enhancements

* Allow `MODULE_LOG_LEVELS` to be read from a config file

---

Use this modular logger approach because:
* Need fine-grained control over log verbosity.
* Want clean, predictable logging output.
