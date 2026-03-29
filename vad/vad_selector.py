# =============================================================================
# vad/vad_selector.py – Factory function for selecting a VAD implementation
# =============================================================================
# Factory pattern: instead of instantiating WebRTCVAD directly everywhere in the
# codebase, callers use get_vad("webrtc") and receive the correct class back.
# Benefit: adding a new VAD backend (e.g. SileroVAD) only requires editing this
# file — run_pipeline.py and all other consumers remain untouched.
# =============================================================================

from vad.webrtc_vad import WebRTCVAD
# SileroVAD is commented out — available for future integration
# from vad.silero_vad import SileroVAD


def get_vad(name: str = "webrtc", **kwargs) -> WebRTCVAD:
    """
    Factory function — returns a VAD instance by name.

    Central point for selecting a VAD implementation. All configuration
    can be passed through **kwargs and will be forwarded to the constructor.

    Args:
        name (str): Name of the VAD backend to use.
                    Currently supported: "webrtc"
                    Planned: "silero"
        **kwargs: Optional parameters forwarded to the VAD constructor.
                  Example: get_vad("webrtc", aggressiveness=3)

    Returns:
        BaseVAD: An instance of the selected VAD class.

    Raises:
        ValueError: If the requested VAD backend is not supported.
    """
    if name == "webrtc":
        # Forward all kwargs directly to the WebRTCVAD constructor.
        # Example: get_vad("webrtc", aggressiveness=1) → WebRTCVAD(aggressiveness=1)
        return WebRTCVAD(**kwargs)

    # SileroVAD — neural network based, more accurate but slower than WebRTC.
    # Uncomment when silero_vad.py has been implemented.
    # elif name == "silero":
    #     return SileroVAD(**kwargs)

    else:
        raise ValueError(
            f"Unsupported VAD type: '{name}'. "
            f"Supported options: 'webrtc'"
        )