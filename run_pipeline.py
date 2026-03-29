import argparse
import collections
import time
import numpy as np
from pathlib import Path

from utils.config import (
    SAMPLE_RATE,
    FRAME_SIZE,
    CHUNK_DURATION_MS,
    VAD_PADDING_MS,
    VAD_TRIGGER_LEVEL,
    MIN_SPEECH_DURATION_SEC,
    MAX_UTTERANCE_SEC,
    DATA_DIR,
)
from utils.logger import setup_logger
from utils.audio_utils import convert_folder
from audio.file_stream import FileStream
from vad.vad_selector import get_vad
from asr.inference import ASRModel
from evaluation.metrics import load_references, aggregate_results
from evaluation.logger import log_results


class BatchVADASR:
    def __init__(self, asr_model: ASRModel, vad, logger):
        self.model = asr_model
        self.vad = vad
        self.logger = logger
        self.num_padding_frames = int(VAD_PADDING_MS / CHUNK_DURATION_MS)
        self.ring_buffer = collections.deque(maxlen=self.num_padding_frames)

    def process_file(self, wav_path: str) -> list[dict]:
        results = []
        stream = FileStream(wav_path)
        triggered = False
        speech_buffer = []
        self.ring_buffer.clear()
        file_time = 0.0
        utterance_start_wall = 0.0

        self.logger.debug(f"Processing: {Path(wav_path).name}")

        for chunk in stream:
            chunk_int16 = (chunk * 32768).astype(np.int16).tobytes()
            is_speech = self.vad.is_speech(chunk_int16, SAMPLE_RATE)
            file_time += CHUNK_DURATION_MS / 1000.0

            if not triggered:
                self.ring_buffer.append((chunk, is_speech))
                num_voiced = sum(1 for _, speech in self.ring_buffer if speech)

                if num_voiced >= VAD_TRIGGER_LEVEL * self.ring_buffer.maxlen:
                    triggered = True
                    utterance_start_wall = time.time()
                    for frame, _ in self.ring_buffer:
                        speech_buffer.append(frame)
                    self.ring_buffer.clear()
                    self.logger.debug(f"  VAD triggered at {file_time:.2f}s")

            else:
                speech_buffer.append(chunk)
                self.ring_buffer.append((chunk, is_speech))
                num_unvoiced = sum(1 for _, speech in self.ring_buffer if not speech)
                current_duration = (len(speech_buffer) * CHUNK_DURATION_MS) / 1000.0

                silence_detected = num_unvoiced >= VAD_TRIGGER_LEVEL * self.ring_buffer.maxlen
                max_duration_hit = current_duration >= MAX_UTTERANCE_SEC

                if silence_detected or max_duration_hit:
                    if max_duration_hit:
                        self.logger.info(f"  Forced flush: utterance exceeded {MAX_UTTERANCE_SEC}s")

                    result = self._flush(speech_buffer, utterance_start_wall)
                    if result:
                        results.append(result)
                        self.logger.info(f"  [{result['duration']:.1f}s] → \"{result['text']}\"")

                    triggered = False
                    speech_buffer = []
                    self.ring_buffer.clear()

        if triggered and speech_buffer:
            result = self._flush(speech_buffer, utterance_start_wall)
            if result:
                results.append(result)

        self.logger.debug(f"  Done: {len(results)} utterances detected")
        return results

    def _flush(self, speech_buffer: list, utterance_start_wall: float) -> dict | None:
        if not speech_buffer:
            return None

        audio = np.concatenate(speech_buffer)
        duration = len(audio) / SAMPLE_RATE

        if duration < MIN_SPEECH_DURATION_SEC:
            return None

        asr_call_start = time.time()
        text = self.model.transcribe(audio)
        asr_call_end = time.time()

        latency_ms = (asr_call_end - asr_call_start) * 1000.0

        if not text or not text.strip():
            return None

        return {
            "text": text,
            "duration": duration,
            "latency": latency_ms,
            "start_time": utterance_start_wall,
            "end_time": asr_call_end,
        }


def run_scenario(scenario_id: int, vad_aggressiveness: int, asr_model: ASRModel, logger) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO {scenario_id} | VAD aggressiveness: {vad_aggressiveness}")
    logger.info(f"{'='*60}")

    raw_dir = Path(DATA_DIR) / "raw" / f"scenario_{scenario_id}"
    processed_dir = Path(DATA_DIR) / "processed" / f"scenario_{scenario_id}"
    ref_file = Path(DATA_DIR) / "references" / f"scenario_{scenario_id}.txt"

    if not raw_dir.exists():
        logger.warning(f"Directory not found: {raw_dir}. Skipping scenario {scenario_id}.")
        return

    logger.info("Converting audio files to 16kHz mono WAV...")
    wav_files = convert_folder(str(raw_dir), str(processed_dir), sample_rate=SAMPLE_RATE)

    if not wav_files:
        logger.warning(f"No audio files for scenario {scenario_id}. Skipping.")
        return

    logger.info(f"Found {len(wav_files)} audio files.")

    references = None
    if ref_file.exists():
        references = load_references(str(ref_file))
        logger.info(f"Loaded {len(references)} reference transcripts.")
        if len(references) != len(wav_files):
            logger.warning(
                f"Reference count ({len(references)}) does not match "
                f"audio file count ({len(wav_files)})."
            )
    else:
        logger.warning(f"No references found at {ref_file}. WER/CER will not be computed.")

    vad = get_vad("webrtc", aggressiveness=vad_aggressiveness)
    processor = BatchVADASR(asr_model=asr_model, vad=vad, logger=logger)

    all_results = []
    for i, wav_path in enumerate(sorted(wav_files)):
        logger.info(f"Processing [{i+1}/{len(wav_files)}]: {Path(wav_path).name}")
        file_results = processor.process_file(wav_path)
        all_results.extend(file_results)

    logger.info(f"\nTotal utterances detected: {len(all_results)}")

    if not all_results:
        logger.warning("No utterances detected. Check audio files and VAD parameters.")
        return

    ref_subset = references[:len(all_results)] if references else None
    metrics = aggregate_results(all_results, references=ref_subset)

    logger.info("\n--- RESULTS ---")
    logger.info(f"  WER:              {metrics.get('wer', 0):.2f}%")
    logger.info(f"  CER:              {metrics.get('cer', 0):.2f}%")
    logger.info(f"  Latency median:   {metrics.get('latency_median', 0):.1f} ms")
    logger.info(f"  Latency P95:      {metrics.get('latency_p95', 0):.1f} ms")
    logger.info(f"  ASR calls/min:    {metrics.get('asr_calls_min', 0):.2f}")
    logger.info(f"  False tokens/s:   {metrics.get('false_tokens', 0):.2f}")

    vad_label = f"webrtc_agr{vad_aggressiveness}"
    log_results(scenario=scenario_id, vad_mode=vad_label, results=metrics)
    logger.info("Results saved to results/results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="WebRTC VAD + Wav2Vec2 ASR batch evaluation")
    parser.add_argument(
        "--scenario", type=int, choices=[1, 2, 3], default=None,
        help="Scenario ID (1=quiet, 2=noisy, 3=variable). Runs all if not specified."
    )
    parser.add_argument(
        "--vad-aggressiveness", type=int, choices=[0, 1, 2, 3], default=2,
        help="WebRTC VAD aggressiveness (0=least, 3=most). Default: 2"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = setup_logger("logs/asr_vad.log")
    logger.info("VAD+ASR Pipeline started")
    logger.info(f"Args: scenario={args.scenario}, vad_aggressiveness={args.vad_aggressiveness}")

    logger.info("Loading ASR model...")
    asr_model = ASRModel()

    scenarios_to_run = [args.scenario] if args.scenario else [1, 2, 3]

    for scenario_id in scenarios_to_run:
        run_scenario(
            scenario_id=scenario_id,
            vad_aggressiveness=args.vad_aggressiveness,
            asr_model=asr_model,
            logger=logger,
        )

    logger.info("\nPipeline complete. Results in: results/results.csv")