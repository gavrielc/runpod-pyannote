"""Speaker diarization handler using pyannote.audio."""

import os
import time
import traceback
import runpod

from .utils import get_memory_usage, get_container_memory_limit, validate_input
from .pipeline_initializer import PipelineSingleton, debug_hook
from .network_utils import download_audio
from .file_utils import cleanup_temp_file

print("=== Starting worker initialization ===")
print(f"Python version: {os.sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print(f"Initial memory usage: {get_memory_usage():.1f} MB")
memory_limit = get_container_memory_limit()
print(f"Container memory limit: {'unlimited' if memory_limit is None else f'{memory_limit:.1f}GB'}")
print("Environment variables:", {k: v for k, v in os.environ.items() if not k.startswith('AWS_')})
print("=== Initialization logs end ===")

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
    audio_path = None
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
        
        # Ensure model is loaded - updated to use singleton
        pipeline = PipelineSingleton.get_pipeline()
        
        # Run diarization
        print("\nStarting diarization...")
        start_time = time.time()
        
        print("Applying pipeline to audio...")
        diarization = pipeline(
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
    finally:
        # Clean up any temporary files
        if audio_path:
            cleanup_temp_file(audio_path)

# Initialize the model when the worker starts
try:
    PipelineSingleton.get_pipeline()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    print(traceback.format_exc())
    raise

runpod.serverless.start({"handler": handler})
