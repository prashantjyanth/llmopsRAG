"""
Simple YAML Loader Utility with Retry Decorator
"""

import yaml
import time
import functools
from pathlib import Path
from datetime import datetime
import uuid


def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator that retries a function call on failure.
    
    Args:
        max_attempts (int): Maximum number of attempts (default: 3)
        delay (float): Initial delay between attempts in seconds (default: 1)
        backoff (float): Multiplier for delay after each attempt (default: 2)
        exceptions (tuple): Tuple of exceptions to catch and retry on (default: (Exception,))
    
    Returns:
        function: Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise e
                    
                    print(f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
        return wrapper
    return decorator


def load_yaml(file_path):
    """
    Simple function to load a YAML file.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Parsed YAML content
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def generate_booking_id(prefix="BK"):
    """
    Generate a unique booking ID using timestamp and UUID.
    
    Args:
        prefix (str): Prefix for the booking ID (default: "BK")
        
    Returns:
        str: Unique booking ID in format: {prefix}{timestamp}{uuid_suffix}
    """
    # Get current timestamp (unix timestamp as integer)
    timestamp = int(time.time())
    
    # Generate 8-character UUID suffix
    uuid_suffix = str(uuid.uuid4()).replace('-', '')[:8].upper()
    
    # Combine prefix, timestamp, and UUID suffix
    booking_id = f"{prefix}{timestamp}{uuid_suffix}"
    
    return booking_id


def generate_simple_booking_id(prefix="BK"):
    """
    Generate a simple booking ID using only timestamp.
    
    Args:
        prefix (str): Prefix for the booking ID (default: "BK")
        
    Returns:
        str: Simple booking ID in format: {prefix}{timestamp}
    """
    timestamp = int(time.time())
    return f"{prefix}{timestamp}"


def generate_readable_booking_id(prefix="BK"):
    """
    Generate a human-readable booking ID using formatted datetime.
    
    Args:
        prefix (str): Prefix for the booking ID (default: "BK")
        
    Returns:
        str: Readable booking ID in format: {prefix}YYYYMMDDHHMMSS{random}
    """
    # Get current datetime
    now = datetime.now()
    date_str = now.strftime("%Y%m%d%H%M%S")
    
    # Add 4-character random suffix
    random_suffix = str(uuid.uuid4()).replace('-', '')[:4].upper()
    
    return f"{prefix}{date_str}{random_suffix}"
    



if __name__ == "__main__":
    # Example usage of booking ID generators
    print("=== Booking ID Generators ===")
    
    # Standard booking ID with timestamp + UUID
    booking_id1 = generate_booking_id()
    print(f"Standard booking ID: {booking_id1}")
    
    # Simple booking ID with only timestamp
    booking_id2 = generate_simple_booking_id()
    print(f"Simple booking ID: {booking_id2}")
    
    # Readable booking ID with formatted date
    booking_id3 = generate_readable_booking_id()
    print(f"Readable booking ID: {booking_id3}")
    
    # Custom prefix examples
    flight_booking = generate_booking_id("FL")
    hotel_booking = generate_booking_id("HT")
    print(f"Flight booking ID: {flight_booking}")
    print(f"Hotel booking ID: {hotel_booking}")
    
    pass