import time
from functools import wraps


def monitor_prediction_time():
    """
    Decorator to measure and log the time taken for a function to execute.

    Returns:
        function: Wrapped function with monitoring logic.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
            finally:
                # Calculate and log the execution time
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Execution time for {func.name}: {execution_time:.4f} seconds")

            return result

        return wrapper

    return decorator