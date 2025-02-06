"""Utilities for audio format detection and conversion."""

import time
import ffmpeg
import traceback
from pathlib import Path

from src.file_utils import temporary_audio_file, cleanup_temp_file

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

def convert_to_wav(input_path):
    """Convert audio file to WAV format."""
    with temporary_audio_file(prefix='converted_', suffix='.wav') as output_path:
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
                str(output_path),
                acodec='pcm_s16le',  # 16-bit PCM
                ac=1,                # mono
                ar='16000',          # 16kHz
                loglevel='warning'   # Show warnings and errors
            )
            
            # Run the conversion with overwrite flag
            out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            
            # Verify the output file exists and is not empty
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Conversion failed: output file is missing or empty")
                
            duration = time.time() - start_time
            print(f"Audio conversion completed in {duration:.2f} seconds")
            
            # Verify the converted file
            output_probe = ffmpeg.probe(str(output_path))
            output_stream = next((stream for stream in output_probe['streams'] 
                                if stream['codec_type'] == 'audio'), None)
            
            if output_stream:
                print(f"Converted audio: {output_stream['codec_name']}, "
                      f"{output_stream['sample_rate']}Hz, "
                      f"{output_stream['channels']} channel(s)")
            
            # Clean up the input file since we're done with it
            cleanup_temp_file(input_path)
            
            return str(output_path)
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            print(f"FFmpeg error: {error_message}")
            raise RuntimeError(f"Failed to convert audio: {error_message}")
        except Exception as e:
            print(f"Unexpected error in convert_to_wav: {str(e)}")
            print(traceback.format_exc())
            raise 