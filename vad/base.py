# =============================================================================
# vad/base.py – Abstract base class for all VAD implementations
# =============================================================================
# Why an abstract base class? It allows any VAD backend (WebRTC, Silero, etc.)
# to be swapped in without changing the rest of the codebase — every VAD class
# guarantees the same interface by inheriting from this base.
# =============================================================================


class BaseVAD:
    """
    Abstract base class for Voice Activity Detection.

    All concrete VAD implementations (WebRTCVAD, SileroVAD, etc.) must
    inherit from this class and implement the is_speech() method.
    """

    def is_speech(self, audio_chunk: bytes, sample_rate: int) -> bool:
        """
        Determines whether a given audio chunk contains speech.

        Must be overridden in every subclass.

        Args:
            audio_chunk (bytes): Raw audio data as a bytes object.
                                 WebRTC VAD requires exactly 10/20/30ms of audio.
            sample_rate (int): Sample rate in Hz.
                               WebRTC VAD supports: 8000, 16000, 32000, 48000.

        Returns:
            bool: True if the chunk contains speech, False if silence or noise.

        Raises:
            NotImplementedError: Always, because this method must be overridden.
        """
        raise NotImplementedError("is_speech() must be implemented in subclasses.")