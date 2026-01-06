"""Utility functions for the project.

This module contains helper functions for logging configuration and device selection.
"""

import logging
import sys
import torch
from typing import Optional

def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger with a standard format.

    Args:
        name (str): The name of the logger.
        level (int): The logging level.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handler exists to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def get_device(device_name: Optional[str] = None) -> torch.device:
    """Returns the appropriate torch device.

    Args:
        device_name (Optional[str]): specific device name to request (e.g., 'cpu', 'cuda:0').
                                     If None, automatically selects cuda or cpu.

    Returns:
        torch.device: The selected device.
    """
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
