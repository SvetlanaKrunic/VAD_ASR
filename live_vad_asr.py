import collections
import sys
import time
import numpy as np

from utils.config import (
    SAMPLE_RATE,
    FRAME_SIZE,
    CHUNK_DURATION_MS,
    VAD_PADDING_MS,
    VAD_TRIGGER_LEVEL,
    MIN_SPEECH_DURATION_SEC,
    MAX_UTTERANCE_SEC,
)
from audio.mic_stream import MicrophoneStream
from vad.webrtc_vad import WebRTCVAD
from asr.inference import ASRModel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class LiveVADASR:
    def __init__(self):
        self.stream = MicrophoneStream(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)
        self.model = ASRModel()
        self.vad = WebRTCVAD()

        self.num_padding_frames = int(VAD_PADDING_MS / CHUNK_DURATION_MS)
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)

        self.triggered = False
        self.buffer = []
        self.results = []

    def run(self, duration: int = 60):
        print(f"\n>>> SYSTEM READY (Padding: {VAD_PADDING_MS}ms, Trigger: {VAD_TRIGGER_LEVEL})")
        print(">>> Speak into the microphone. Press Ctrl+C to stop.\n")

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                chunk_np = self.stream.read_chunk()
                if chunk_np is not None:
                    self._process_chunk(chunk_np)
        except KeyboardInterrupt:
            print("\n>>> Stopped by user.")
        finally:
            self.stream.close()

        return self.results

    def _process_chunk(self, chunk_np: np.ndarray):
        chunk_int16 = (chunk_np * 32768).astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(chunk_int16, SAMPLE_RATE)

        if not self.triggered:
            self.ring_buffer.append((chunk_np, is_speech))
            num_voiced = sum(1 for _, speech in self.ring_buffer if speech)

            if num_voiced >= VAD_TRIGGER_LEVEL * self.ring_buffer.maxlen:
                self.triggered = True
                for frame, _ in self.ring_buffer:
                    self.buffer.append(frame)
                self.ring_buffer.clear()

        else:
            self.buffer.append(chunk_np)
            self.ring_buffer.append((chunk_np, is_speech))
            num_unvoiced = sum(1 for _, speech in self.ring_buffer if not speech)
            current_duration = (len(self.buffer) * CHUNK_DURATION_MS) / 1000.0

            silence_detected = num_unvoiced >= VAD_TRIGGER_LEVEL * self.ring_buffer.maxlen
            max_duration_hit = current_duration >= MAX_UTTERANCE_SEC

            if silence_detected or max_duration_hit:
                if max_duration_hit:
                    print(f"[INFO] Forced flush: utterance exceeded {MAX_UTTERANCE_SEC}s")

                self.triggered = False
                self._flush_buffer()
                self.ring_buffer.clear()

    def _flush_buffer(self):
        if not self.buffer:
            return

        audio = np.concatenate(self.buffer)
        duration = len(audio) / SAMPLE_RATE
        self.buffer = []

        if duration < MIN_SPEECH_DURATION_SEC:
            return

        text = self.model.transcribe(audio)
        if text and text.strip():
            print(f"[{time.strftime('%H:%M:%S')}] TRANSCRIBED ({duration:.1f}s): {text}")
            self.results.append({"text": text, "duration": duration})


if __name__ == "__main__":
    asr_system = LiveVADASR()
    asr_system.run(duration=60)