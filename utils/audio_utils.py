import subprocess
import os
import numpy as np
import wave
from pathlib import Path


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000, mono: bool = True) -> None:
    """
    Converts any audio file (MP3, FLAC, OGG, etc.) to WAV format using ffmpeg.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path where the output WAV file will be saved.
        sample_rate (int): Target sample rate in Hz. Default 16000 because
                           Wav2Vec2 and WebRTC VAD both require exactly 16kHz.
        mono (bool): If True, convert to mono (1 channel). ASR models expect mono.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If ffmpeg conversion fails.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build the ffmpeg command as a list of arguments (safer than a shell string)
    command = [
        "ffmpeg",
        "-y",              # -y = overwrite output file if it already exists
        "-i", input_path,  # -i = input file
        "-ar", str(sample_rate),       # -ar = target audio sample rate
        "-ac", "1" if mono else "2",   # -ac = number of channels (1=mono, 2=stereo)
        output_path
    ]

    try:
        # check=True raises an exception if ffmpeg returns a non-zero exit code
        # stdout/stderr=PIPE captures output instead of printing it to the terminal
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode().strip()}")


def convert_folder(input_dir: str, output_dir: str, sample_rate: int = 16000) -> list[str]:
    """
    Converts all MP3 and WAV files in a folder to 16kHz mono WAV.

    Used for batch data preparation before running the evaluation pipeline.

    Args:
        input_dir (str): Folder containing the original audio files.
        output_dir (str): Folder where converted WAV files will be saved.
        sample_rate (int): Target sample rate for all output files.

    Returns:
        list[str]: List of paths to all successfully converted WAV files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all MP3 and WAV files from the input folder
    audio_files = list(input_path.glob("*.mp3")) + list(input_path.glob("*.wav"))

    if not audio_files:
        print(f"[WARNING] No audio files found in: {input_dir}")
        return []

    converted = []
    for audio_file in sorted(audio_files):
        # Output file always has .wav extension regardless of input format
        out_file = output_path / (audio_file.stem + ".wav")

        print(f"  Converting: {audio_file.name} → {out_file.name}")
        try:
            convert_to_wav(str(audio_file), str(out_file), sample_rate=sample_rate)
            converted.append(str(out_file))
        except Exception as e:
            print(f"  [ERROR] {audio_file.name}: {e}")

    return converted


def load_wav(wav_path: str) -> tuple:
    """
    Loads a WAV file and returns a float32 numpy array and the sample rate.

    The Wav2Vec2 processor expects audio as a float32 array normalized to [-1.0, 1.0].
    This function performs that normalization automatically.

    Args:
        wav_path (str): Path to the WAV file.

    Returns:
        tuple: (numpy float32 array, sample_rate int)

    Raises:
        FileNotFoundError: If the WAV file does not exist.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(wav_path, 'rb') as wf:
        # Read all frames from the WAV file as raw bytes
        raw_bytes = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()

    # Convert raw bytes to a numpy array
    # dtype=int16 because standard WAV files store 16-bit signed integers
    audio = np.frombuffer(raw_bytes, dtype=np.int16)

    # If the file is stereo (2 channels), keep only the left channel
    if n_channels == 2:
        audio = audio[::2]

    # Normalize to float32 in the range [-1.0, 1.0]
    # 32768 = 2^15 = maximum value of a signed 16-bit integer
    audio = audio.astype(np.float32) / 32768.0

    return audio, sample_rate