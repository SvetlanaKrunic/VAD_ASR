# =============================================================================
# vad/webrtc_vad.py – WebRTC VAD implementation
# =============================================================================
# WebRTC VAD is Google's open-source voice activity detector, originally built
# for the WebRTC protocol (browser-based audio/video calls). It was chosen because:
#   - It is extremely fast (C implementation underneath; Python is just a thin wrapper)
#   - It is deterministic — identical input always produces identical output
#   - It requires no neural network — no GPU needed, runs fine on a Raspberry Pi
#   - It is industry-proven, shipped inside Chrome, Firefox, and countless other products
#
# Installation: pip install webrtcvad
# GitHub: https://github.com/wiseman/py-webrtcvad
# =============================================================================

import webrtcvad
from vad.base import BaseVAD
from utils.config import VAD_AGGRESSIVENESS, SAMPLE_RATE


class WebRTCVAD(BaseVAD):
    """
    WebRTC-based Voice Activity Detector.

    Inherits from BaseVAD and implements is_speech() using Google's webrtcvad package.
    """

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        """
        Initializes the WebRTC VAD with the given aggressiveness level.

        Args:
            aggressiveness (int): Noise filtering aggressiveness.
                                  Range: 0 (least aggressive) to 3 (most aggressive).
                                  Default is taken from utils/config.py.
        """
        # Instantiate the WebRTC VAD object — this initializes the underlying C library
        self.vad = webrtcvad.Vad()

        # Set the aggressiveness mode — must be an integer between 0 and 3
        # Higher value = more aggressive noise filtering, but greater risk of
        # clipping quiet speech or the ends of words
        self.vad.set_mode(aggressiveness)

    def is_speech(self, audio_chunk: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """
        Determines whether a given audio chunk contains speech.

        WebRTC VAD internally uses a Gaussian Mixture Model (GMM) and
        statistical methods to distinguish speech from background noise.

        IMPORTANT — WebRTC VAD has strict input requirements:
          - audio_chunk must represent exactly 10ms, 20ms, or 30ms of audio
          - sample_rate must be 8000, 16000, 32000, or 48000 Hz
          - format: 16-bit signed PCM (little-endian bytes)
          - byte length = (sample_rate * duration_ms / 1000) * 2

        Args:
            audio_chunk (bytes): Audio window as raw PCM bytes (int16, little-endian).
            sample_rate (int): Sample rate in Hz. Default taken from config.py.

        Returns:
            bool: True = speech detected, False = silence or noise.
        """
        try:
            # Directly invoke the WebRTC C implementation.
            # When the input is valid, this completes in microseconds.
            return self.vad.is_speech(audio_chunk, sample_rate)
        except Exception as e:
            # Most common cause of error: incorrect chunk size.
            # Log and return False so the pipeline can continue processing.
            print(f"[VAD Error] Invalid audio chunk: {e}")
            return False