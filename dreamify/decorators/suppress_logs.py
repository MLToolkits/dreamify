import functools
import os
import sys


def suppress_logs(func):
    """Decorator to suppress stdout and stderr during function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as fnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = fnull, fnull
                return func(*args, **kwargs)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    return wrapper
