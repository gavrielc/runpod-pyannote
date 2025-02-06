"""Pipeline initialization and model loading utilities."""

import os
import time
import torch
import traceback
from threading import Lock
from pyannote.audio import Pipeline
from .utils import get_memory_usage

class PipelineSingleton:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_pipeline(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check pattern
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
                        pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=auth_token
                        )
                        
                        # Configure based on available hardware
                        if torch.cuda.is_available():
                            print("CUDA is available, optimizing for GPU")
                            pipeline.to(torch.device("cuda"))
                            torch.backends.cudnn.benchmark = True
                            print(f"Using GPU: {torch.cuda.get_device_name()}")
                            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                        else:
                            print("CUDA is not available, running on CPU")
                            
                        duration = time.time() - start_time
                        print(f"Pipeline loaded in {duration:.2f} seconds")
                        cls._instance = pipeline
                        
                    except Exception as e:
                        print(f"Error in pipeline initialization: {str(e)}")
                        print(traceback.format_exc())
                        raise
        return cls._instance

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

def load_model():
    """Load the pyannote.audio pipeline."""
    try:
        return PipelineSingleton.get_pipeline()
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        print(traceback.format_exc())
        raise 