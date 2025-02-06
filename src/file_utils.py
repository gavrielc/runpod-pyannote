"""Utilities for handling temporary files and file operations."""

import os
import time
import uuid
import tempfile
import contextlib
from pathlib import Path
from urllib.parse import urlparse, unquote

def get_file_extension_from_url(url):
    """Extract file extension from URL, defaulting to .wav if none found."""
    path = unquote(urlparse(url).path)
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else '.wav'

def generate_unique_filename(prefix, extension):
    """Generate a unique filename using timestamp and random string."""
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join('/tmp', f"{prefix}_{timestamp}_{unique_id}{extension}")

@contextlib.contextmanager
def temporary_audio_file(prefix, suffix):
    """Context manager for temporary audio files without auto deletion.
    Caller is responsible for cleanup using cleanup_temp_file()."""
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    try:
        yield Path(temp_file.name)
    finally:
        temp_file.close()

def cleanup_temp_file(file_path):
    """Safely clean up a temporary file.
    
    Args:
        file_path: String or Path object pointing to the file to remove
    """
    if file_path:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary file {file_path}: {e}") 