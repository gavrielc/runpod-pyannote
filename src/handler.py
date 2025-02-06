""" Speaker diarization handler using pyannote.audio. """

import os
import time
import torch
import ffmpeg
import runpod
import psutil
import traceback
from pyannote.audio import Pipeline
import requests
from urllib.parse import urlparse, unquote
import uuid

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

print("=== Starting worker initialization ===")
print(f"Python version: {os.sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print(f"Initial memory usage: {get_memory_usage():.1f} MB")
memory_limit = get_container_memory_limit()
print(f"Container memory limit: {'unlimited' if memory_limit is None else f'{memory_limit:.1f}GB'}")
print("Environment variables:", {k: v for k, v in os.environ.items() if not k.startswith('AWS_')})
print("=== Initialization logs end ===")

# Load the pipeline globally for reuse across requests
PIPELINE = None

def get_file_extension_from_url(url):
    """Extract file extension from URL, defaulting to .wav if none found."""
    path = unquote(urlparse(url).path)
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else '.wav'

def detect_audio_format(file_path):
    """Detect audio format using ffprobe."""
    try:
        probe = ffmpeg.probe(file_path)
        if 'streams' in probe and len(probe['streams']) > 0:
            for stream in probe['streams']:
                if stream['codec_type'] == 'audio':
                    return stream['codec_name']
        return None
    except ffmpeg.Error:
        return None

def generate_unique_filename(prefix, extension):
    """Generate a unique filename using timestamp and random string."""
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join('/tmp', f"{prefix}_{timestamp}_{unique_id}{extension}")

def download_audio(url):
    """Download audio from pre-signed URL with proper format handling."""
    try:
        # Get extension from URL
        extension = get_file_extension_from_url(url)
        temp_path = generate_unique_filename('input', extension)
        print(f"Downloading audio from URL (detected format: {extension})")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        
        if file_size == 0:
            print("Warning: Content-Length header not available, cannot track progress")
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
                    
                    # Log progress every second
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        if file_size:
                            progress = (bytes_downloaded / file_size) * 100
                            speed = bytes_downloaded / (current_time - start_time) / 1024 / 1024  # MB/s
                            print(f"Download progress: {progress:.1f}% ({speed:.1f}MB/s)")
                        else:
                            speed = bytes_downloaded / (current_time - start_time) / 1024 / 1024  # MB/s
                            print(f"Downloaded {bytes_downloaded / 1024 / 1024:.1f}MB ({speed:.1f}MB/s)")
                        last_log_time = current_time
        
        total_time = time.time() - start_time
        average_speed = bytes_downloaded / total_time / 1024 / 1024  # MB/s
        print(f"Download completed: {bytes_downloaded / 1024 / 1024:.1f}MB in {total_time:.1f}s ({average_speed:.1f}MB/s)")
        
        # Detect actual format
        detected_format = detect_audio_format(temp_path)
        print(f"Detected audio format: {detected_format}")
        
        # Convert to WAV if needed
        if detected_format and detected_format != 'pcm_s16le':
            print(f"Converting from {detected_format} to WAV format...")
            wav_path = convert_to_wav(temp_path)
            try:
                os.remove(temp_path)  # Clean up original file
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {e}")
            return wav_path
        elif detected_format == 'pcm_s16le':
            print("File is already in WAV format, no conversion needed")
            return temp_path
        else:
            print("Could not detect format, attempting conversion anyway...")
            wav_path = convert_to_wav(temp_path)
            try:
                os.remove(temp_path)  # Clean up original file
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {e}")
            return wav_path
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download audio: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during download: {str(e)}")

def convert_to_wav(input_path):
    """Convert audio file to WAV format."""
    output_path = generate_unique_filename('converted', '.wav')
    try:
        print(f"Starting audio conversion of {input_path}")
        start_time = time.time()
        
        # Get input file information
        probe = ffmpeg.probe(input_path)
        input_sample_rate = next(
            (stream['sample_rate'] for stream in probe['streams'] 
             if stream['codec_type'] == 'audio'),
            '16000'  # default to 16kHz if not found
        )
        
        print(f"Input sample rate: {input_sample_rate}Hz")
        
        # Convert to WAV with specific parameters
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(
            stream, 
            output_path,
            acodec='pcm_s16le',  # 16-bit PCM
            ac=1,                # mono
            ar='16000',          # 16kHz
            loglevel='warning'   # Show warnings and errors
        )
        
        # Run the conversion
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        
        # Verify the output file exists and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Conversion failed: output file is missing or empty")
            
        duration = time.time() - start_time
        print(f"Audio conversion completed in {duration:.2f} seconds")
        
        # Verify the converted file
        output_probe = ffmpeg.probe(output_path)
        output_stream = next((stream for stream in output_probe['streams'] 
                            if stream['codec_type'] == 'audio'), None)
        
        if output_stream:
            print(f"Converted audio: {output_stream['codec_name']}, "
                  f"{output_stream['sample_rate']}Hz, "
                  f"{output_stream['channels']} channel(s)")
        
        return output_path
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print(f"FFmpeg error: {error_message}")
        raise RuntimeError(f"Failed to convert audio: {error_message}")
    except Exception as e:
        print(f"Unexpected error in convert_to_wav: {str(e)}")
        print(traceback.format_exc())
        raise

def get_optimal_batch_size():
    """Determine optimal batch size based on available hardware."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_mem >= 40:  # A100, A6000, etc
            return 64
        elif gpu_mem >= 20:  # A4500, A5000
            return 32
        else:  # Smaller GPUs
            return 16
    else:
        # For CPU, use smaller batch size
        return 8

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

def load_model():
    global PIPELINE
    if PIPELINE is None:
        try:
            print("Loading pyannote.audio pipeline...")
            start_time = time.time()
            
            auth_token = os.environ.get('HUGGINGFACE_TOKEN')
            if not auth_token:
                raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
            
            print("Environment check:")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Python version: {os.sys.version}")
            print(f"Torch version: {torch.__version__}")
            print(f"Number of CPU cores: {os.cpu_count()}")
            
            # Load pipeline with default settings
            PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            
            # Configure based on available hardware
            if torch.cuda.is_available():
                print("CUDA is available, optimizing for GPU")
                PIPELINE.to(torch.device("cuda"))
                # Enable cudnn benchmarking for better performance
                torch.backends.cudnn.benchmark = True
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("CUDA is not available, running on CPU")
                
            duration = time.time() - start_time
            print(f"Pipeline loaded in {duration:.2f} seconds")
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            print(traceback.format_exc())
            raise

def debug_hook(step_name, step_artifact, file=None, **kwargs):
    """Debug hook for pyannote pipeline progress."""
    current_time = time.strftime('%H:%M:%S')
    print(f"\nStep: {step_name} (Time: {current_time})")
    
    if step_name == "embeddings":
        # Print detailed GPU info during embeddings step
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            print(f"GPU Memory cached: {torch.cuda.memory_cached() / 1024**2:.1f}MB")
            # Try to force garbage collection
            torch.cuda.empty_cache()
    
    if 'completed' in kwargs and 'total' in kwargs:
        completed = kwargs['completed']
        total = kwargs['total']
        percent = (completed / total) * 100 if total > 0 else 0
        print(f"Progress: {completed}/{total} ({percent:.1f}%)")
    
    print(f"CPU Memory usage: {get_memory_usage():.1f} MB")
    
    # Print the type and shape of the artifact if available
    if step_artifact is not None:
        if isinstance(step_artifact, torch.Tensor):
            print(f"Artifact shape: {step_artifact.shape}")
            print(f"Artifact device: {step_artifact.device}")
            print(f"Artifact dtype: {step_artifact.dtype}")
        else:
            print(f"Artifact type: {type(step_artifact)}")

def handler(job):
    """Handler for speaker diarization requests.
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://storage-service.com/audio/123.wav",  # Pre-signed URL
            "options": {
                "num_speakers": null,      # optional
                "min_speakers": null,      # optional
                "max_speakers": null,      # optional
                "debug": false            # optional
            }
        }
    }
    """
    try:
        print(f"\nReceived job input: {job}")
        job_input = job['input']
        
        # Validate input
        validate_input(job_input)
        
        # Get options with defaults
        options = job_input.get('options', {})
        num_speakers = options.get('num_speakers')
        min_speakers = options.get('min_speakers')
        max_speakers = options.get('max_speakers')
        debug_mode = options.get('debug', False)
        
        print(f"Processing with options: {options}")
        
        # Download and convert audio if needed
        audio_path = download_audio(job_input['audio_url'])
        print(f"Final audio path: {audio_path}")
        
        # Ensure model is loaded
        if PIPELINE is None:
            print("Loading model...")
            load_model()
            print("Model loaded")
        
        # Run diarization
        print("\nStarting diarization...")
        start_time = time.time()
        
        print("Applying pipeline to audio...")
        diarization = PIPELINE(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hook=debug_hook if debug_mode else None
        )
        
        # Convert diarization results to a list of segments
        print("Converting results to segments...")
        results = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end
            })
            if len(results) % 100 == 0:
                print(f"Processed {len(results)} segments...")
        
        duration = time.time() - start_time
        
        # Clean up
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary file: {e}")
        
        return {
            "segments": results,
            "metadata": {
                "processing_time_seconds": duration,
                "num_speakers": len(set(r['speaker'] for r in results)),
                "num_segments": len(results)
            }
        }
        
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

# Initialize the model when the worker starts
try:
    load_model()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    print(traceback.format_exc())
    raise

runpod.serverless.start({"handler": handler})
