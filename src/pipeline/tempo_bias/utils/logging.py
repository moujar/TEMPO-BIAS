import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = None, level=logging.INFO):
    """Configure logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"run_{timestamp}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
