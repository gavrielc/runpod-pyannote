"""Utilities for handling network operations and downloads."""

import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pathlib import Path

from .file_utils import temporary_audio_file, get_file_extension_from_url
from .audio_conversion import detect_audio_format, convert_to_wav

def create_robust_session():
    """Create a requests session with retry logic and timeouts."""
    session = requests.Session()
    
    # Configure retry strategy
    retries = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["GET"]  # only retry on GET requests
    )
    
    # Add retry adapter to session
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def download_audio(url):
    """Download audio from pre-signed URL with proper format handling."""
    extension = get_file_extension_from_url(url)
    
    with temporary_audio_file(prefix='input_', suffix=extension) as temp_path:
        try:
            print(f"Downloading audio from URL (detected format: {extension})")
            
            session = create_robust_session()
            # Add timeouts: (connect timeout, read timeout)
            with session.get(url, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                
                file_size = int(response.headers.get('content-length', 0))
                
                if file_size == 0:
                    print("Warning: Content-Length header not available")
                else:
                    print(f"File size: {file_size / 1024 / 1024:.1f}MB")
                
                start_time = time.time()
                bytes_downloaded = 0
                last_log_time = start_time
                
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            bytes_downloaded += len(chunk)
                            f.write(chunk)
                            
                            current_time = time.time()
                            if current_time - last_log_time >= 1.0:
                                if file_size:
                                    progress = (bytes_downloaded / file_size) * 100
                                    speed = bytes_downloaded / (current_time - start_time) / 1024 / 1024
                                    print(f"Download progress: {progress:.1f}% ({speed:.1f}MB/s)")
                                else:
                                    speed = bytes_downloaded / (current_time - start_time) / 1024 / 1024
                                    print(f"Downloaded {bytes_downloaded / 1024 / 1024:.1f}MB ({speed:.1f}MB/s)")
                                last_log_time = current_time
                
                total_time = time.time() - start_time
                average_speed = bytes_downloaded / total_time / 1024 / 1024
                print(f"Download completed: {bytes_downloaded / 1024 / 1024:.1f}MB in {total_time:.1f}s ({average_speed:.1f}MB/s)")
            
            detected_format = detect_audio_format(str(temp_path))
            print(f"Detected audio format: {detected_format}")
            
            if detected_format and detected_format != 'pcm_s16le':
                print(f"Converting from {detected_format} to WAV format...")
                return convert_to_wav(str(temp_path))
            elif detected_format == 'pcm_s16le':
                print("File is already in WAV format, no conversion needed")
                return str(temp_path)
            else:
                print("Could not detect format, attempting conversion anyway...")
                return convert_to_wav(str(temp_path))
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download audio: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during download: {str(e)}") 