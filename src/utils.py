"""General utility functions."""

import os
import psutil

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def get_container_memory_limit():
    """Get the container's memory limit in GB."""
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit = int(f.read().strip()) / (1024**3)  # Convert bytes to GB
            if limit > 1000000:  # If > 1000000GB, probably no limit
                return None
            return limit
    except:
        try:
            # Try v2 cgroup path
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                content = f.read().strip()
                if content == 'max':
                    return None
                return int(content) / (1024**3)  # Convert bytes to GB
        except:
            return None

def validate_input(job_input):
    """Validate the input parameters."""
    if not job_input.get('audio_url'):
        raise ValueError("audio_url is required")
    
    options = job_input.get('options', {})
    if not isinstance(options, dict):
        raise ValueError("options must be a dictionary")
    
    # Validate speaker parameters
    num_speakers = options.get('num_speakers')
    if num_speakers is not None and not isinstance(num_speakers, int):
        raise ValueError("num_speakers must be an integer")
    
    min_speakers = options.get('min_speakers')
    if min_speakers is not None and not isinstance(min_speakers, int):
        raise ValueError("min_speakers must be an integer")
    
    max_speakers = options.get('max_speakers')
    if max_speakers is not None and not isinstance(max_speakers, int):
        raise ValueError("max_speakers must be an integer")
    
    return True 