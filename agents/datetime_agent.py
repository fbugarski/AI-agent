"""Agent for handling datetime operations."""

from datetime import datetime

def get_current_datetime():
    """
    Get the current date and time.
    
    Returns:
        str: The current date and time as a string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
