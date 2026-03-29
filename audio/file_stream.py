# =============================================================================
# audio/file_stream.py – Reads a WAV file chunk by chunk, simulating a microphone
# =============================================================================
# Why is this needed? WebRTC VAD cannot process an entire audio file at once —
# it must receive small windows (10/20/30ms) one at a time, exactly as it would
# from a live microphone. This class simulates that streaming behaviour over a
# pre-recorded file, allowing the same VAD+ASR pipeline to run in both
# live and batch evaluation modes without any changes to the core logic.
# =============================================================================

import numpy as np
from utils.config import SAMPLE_RATE, FRAME_SIZE
from utils.audio_utils import load_wav


class FileStream:
    """
    Iterator that loads a WAV file and yields it chunk by chunk.

    Mirrors the interface of MicrophoneStream so that the VAD+ASR pipeline
    works identically for both live input and recorded audio files.
    """

    def __init__(self, wav_path: str, sample_rate: int = SAMPLE_RATE, frame_size: int = FRAME_SIZE):
        """
        Args:
            wav_path (str): Path to the WAV file (must be 16kHz mono).
            sample_rate (int): Expected sample rate — must match the file.
            frame_size (int): Number of samples per chunk, from config.py.
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        # Load the entire audio file into memory as a float32 array
        self.audio, file_sr = load_wav(wav_path)

        # Verify the file's sample rate matches the expected configuration
        if file_sr != sample_rate:
            raise ValueError(
                f"File sample rate ({file_sr} Hz) does not match config ({sample_rate} Hz). "
                f"Run audio_utils.convert_to_wav() first."
            )

        # Cursor tracks the current read position in the audio array
        self.cursor = 0

        # Chunk duration in ms — used for logging and duration calculations
        self.chunk_duration_ms = (frame_size / sample_rate) * 1000

    def __iter__(self):
        # Reset the cursor to the beginning of the file on each new iteration
        self.cursor = 0
        return self

    def __next__(self) -> np.ndarray:
        """
        Returns the next chunk as a float32 numpy array.
        Raises StopIteration when the file is exhausted.
        """
        start = self.cursor
        end = start + self.frame_size

        # Stop iteration when we have passed the end of the file
        if start >= len(self.audio):
            raise StopIteration

        chunk = self.audio[start:end]

        # If the last chunk is shorter than frame_size, pad with zeros (silence)
        # WebRTC VAD requires exactly frame_size samples — no fewer
        if len(chunk) < self.frame_size:
            chunk = np.pad(chunk, (0, self.frame_size - len(chunk)), mode='constant')

        self.cursor = end
        return chunk

    def read_chunk(self) -> np.ndarray | None:
        """
        Alternative read method compatible with the MicrophoneStream interface.
        Returns None when the file is exhausted instead of raising StopIteration.
        """
        try:
            return self.__next__()
        except StopIteration:
            return None

    def total_duration_sec(self) -> float:
        """Returns the total duration of the audio file in seconds."""
        return len(self.audio) / self.sample_rate