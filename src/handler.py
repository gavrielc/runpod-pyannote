""" Speaker diarization handler using pyannote.audio. """

import os
import torch
import runpod
from pyannote.audio import Pipeline

# Load the pipeline globally for reuse across requests
PIPELINE = None

def load_model():
    global PIPELINE
    if PIPELINE is None:
        auth_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not auth_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
        
        PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            PIPELINE.to(torch.device("cuda"))

def handler(job):
    """ Handler function for speaker diarization. """
    try:
        job_input = job['input']
        
        if 'audio_path' not in job_input:
            return {"error": "audio_path is required in the input"}
        
        audio_path = job_input['audio_path']
        
        # Optional parameters
        num_speakers = job_input.get('num_speakers', None)
        min_speakers = job_input.get('min_speakers', None)
        max_speakers = job_input.get('max_speakers', None)
        
        # Ensure model is loaded
        if PIPELINE is None:
            load_model()
        
        # Run diarization
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
        
        return {
            "segments": results
        }
        
    except Exception as e:
        return {"error": str(e)}

# Initialize the model when the worker starts
load_model()

runpod.serverless.start({"handler": handler})
