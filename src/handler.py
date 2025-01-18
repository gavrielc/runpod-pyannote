""" Speaker diarization handler using pyannote.audio. """

import os
import time
import torch
import ffmpeg
import runpod
from pyannote.audio import Pipeline

# Load the pipeline globally for reuse across requests
PIPELINE = None

def convert_to_wav(input_path):
    """Convert audio file to WAV format."""
    output_path = os.path.splitext(input_path)[0] + '.wav'
    try:
        print(f"Starting audio conversion of {input_path}")
        start_time = time.time()
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        duration = time.time() - start_time
        print(f"Audio conversion completed in {duration:.2f} seconds")
        return output_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to convert audio: {e.stderr.decode()}")

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

def load_model():
    global PIPELINE
    if PIPELINE is None:
        print("Loading pyannote.audio pipeline...")
        start_time = time.time()
        
        auth_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not auth_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
        
        # Get optimal parameters based on hardware
        batch_size = get_optimal_batch_size()
        num_workers = os.cpu_count() or 4  # CPU count or default to 4
        if not torch.cuda.is_available():
            # Reduce workers for CPU to avoid memory issues
            num_workers = min(num_workers, 2)
        
        print(f"Using batch_size={batch_size}, num_workers={num_workers}")
        
        PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        # Configure based on available hardware
        if torch.cuda.is_available():
            print("CUDA is available, moving pipeline to GPU")
            PIPELINE.to(torch.device("cuda"))
            # GPU optimizations
            PIPELINE = PIPELINE.half()  # FP16 for GPU
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("CUDA is not available, optimizing for CPU")
            
        duration = time.time() - start_time
        print(f"Pipeline loaded in {duration:.2f} seconds")

def handler(job):
    """ Handler function for speaker diarization. """
    try:
        job_input = job['input']
        
        if 'audio_path' not in job_input:
            return {"error": "audio_path is required in the input"}
        
        audio_path = job_input['audio_path']
        print(f"\nProcessing audio file: {audio_path}")
        
        # Convert audio to WAV format if needed
        if not audio_path.lower().endswith('.wav'):
            audio_path = convert_to_wav(audio_path)
        
        # Optional parameters
        num_speakers = job_input.get('num_speakers', None)
        min_speakers = job_input.get('min_speakers', None)
        max_speakers = job_input.get('max_speakers', None)
        
        # Ensure model is loaded
        if PIPELINE is None:
            load_model()
        
        # Run diarization
        print("\nStarting diarization...")
        start_time = time.time()
        
        diarization = PIPELINE(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Convert diarization results to a list of segments
        results = []
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end
            })
        
        duration = time.time() - start_time
        print(f"\nDiarization completed in {duration:.2f} seconds ({duration/60:.1f} minutes)")
        print(f"Found {len(set(r['speaker'] for r in results))} speakers")
        print(f"Generated {len(results)} segments")
        
        return {
            "segments": results,
            "processing_time": duration,
            "num_speakers": len(set(r['speaker'] for r in results)),
            "num_segments": len(results)
        }
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return {"error": str(e)}

# Initialize the model when the worker starts
load_model()

runpod.serverless.start({"handler": handler})
