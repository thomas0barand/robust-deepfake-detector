import os
import subprocess
from pathlib import Path
from typing import Union, Optional

SUPPORTED_FORMATS = {'.wav', '.m4a', '.mp4', '.flac', '.ogg', '.aac', '.wma', '.mp3'}

def convert(
    input_path: Union[str, Path],
    output_format: str = 'mp3',
    output_dir: Optional[Union[str, Path]] = None,
    bitrate: str = '320k',
    preserve_sample_rate: bool = True
) -> list:
    """
    Convert audio file(s) to specified format without resampling or transforming.
    
    Args:
        input_path: Path to file or directory to convert
        output_format: Target format (default: 'mp3')
        output_dir: Output directory (default: same as input file)
        bitrate: Bitrate for lossy formats (default: '320k' for highest quality)
        preserve_sample_rate: Keep original sample rate (default: True)
    
    Returns:
        List of converted file paths
    """
    input_path = Path(input_path)
    converted_files = []
    
    if input_path.is_file():
        result = _convert_file(input_path, output_format, output_dir, bitrate, preserve_sample_rate)
        if result:
            converted_files.append(result)
    elif input_path.is_dir():
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                result = _convert_file(file_path, output_format, output_dir, bitrate, preserve_sample_rate)
                if result:
                    converted_files.append(result)
    else:
        raise ValueError(f"Invalid path: {input_path}")
    
    return converted_files

def _convert_file(
    input_file: Path,
    output_format: str,
    output_dir: Optional[Path],
    bitrate: str,
    preserve_sample_rate: bool
) -> Optional[Path]:
    """Convert a single audio file using FFmpeg."""
    input_ext = input_file.suffix.lower()
    
    if input_ext not in SUPPORTED_FORMATS:
        print(f"Skipping unsupported format: {input_file}")
        return None
    
    if input_ext == f'.{output_format}':
        print(f"Skipping (already {output_format}): {input_file}")
        return None
    
    # Determine output path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_file.stem}.{output_format}"
    else:
        output_file = input_file.parent / f"{input_file.stem}.{output_format}"
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-i', str(input_file), '-y']
    
    if output_format == 'mp3':
        # High-quality MP3 encoding (CBR for consistency)
        cmd.extend([
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            '-write_xing', '0'  # Disable VBR for consistent bitrate
        ])
    elif output_format == 'wav':
        # Lossless WAV (preserves exact audio)
        cmd.extend(['-codec:a', 'pcm_s16le'])
    elif output_format == 'flac':
        # Lossless FLAC (preserves exact audio with compression)
        cmd.extend(['-codec:a', 'flac', '-compression_level', '8'])
    else:
        # Default encoding
        cmd.extend(['-b:a', bitrate])
    
    # Preserve metadata
    cmd.extend(['-map_metadata', '0'])
    
    cmd.append(str(output_file))
    
    try:
        print(f"Converting: {input_file.name} -> {output_file.name}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✓ Success: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {input_file}: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        return None

def get_audio_info(file_path: Union[str, Path]) -> dict:
    """Get audio file information using FFprobe."""
    file_path = Path(file_path)
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        audio_stream = next(
            (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
            {}
        )
        
        return {
            'format': data.get('format', {}).get('format_name', 'unknown'),
            'duration': float(data.get('format', {}).get('duration', 0)),
            'sample_rate': int(audio_stream.get('sample_rate', 0)),
            'channels': int(audio_stream.get('channels', 0)),
            'codec': audio_stream.get('codec_name', 'unknown'),
            'bitrate': int(audio_stream.get('bit_rate', 0))
        }
    except Exception as e:
        print(f"Error getting audio info: {e}")
        return {}

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python utils.py <input_file_or_folder> [output_format] [bitrate]")
        print("Example: python utils.py ../data/ai/ mp3 320k")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'mp3'
    bitrate = sys.argv[3] if len(sys.argv) > 3 else '320k'
    
    print(f"Converting to {output_format} format at {bitrate} bitrate...")
    converted = convert(input_path, output_format=output_format, bitrate=bitrate)
    print(f"\nConverted {len(converted)} file(s)")
