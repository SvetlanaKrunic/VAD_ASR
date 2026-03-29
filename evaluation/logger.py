# =============================================================================
# evaluation/logger.py – Saving evaluation results to a CSV file
# =============================================================================
# Each experiment (scenario + VAD configuration) is logged as a single row
# in a CSV table. This makes it easy to compare results across runs in
# Excel, pandas, or any other data analysis tool.
# =============================================================================

import csv
from pathlib import Path
from datetime import datetime
from utils.config import RESULTS_DIR

# Path to the output CSV file — created automatically on the first call
OUT_FILE = Path(RESULTS_DIR) / "results.csv"

# CSV column headers — the order here must match the order in log_results()
HEADER = [
    "Timestamp",           # When the experiment was run
    "Scenario",            # Scenario ID (1=quiet, 2=noisy, 3=variable rhythm)
    "VAD",                 # VAD aggressiveness label
    "Latency median (ms)", # Median ASR response latency
    "Latency p95 (ms)",    # 95th percentile latency (worst case for 95% of utterances)
    "ASR calls/min",       # System throughput — how many utterances processed per minute
    "False tokens/s",      # Hallucinated words per second
    "WER (%)",             # Word Error Rate
    "CER (%)",             # Character Error Rate
]


def _write_header() -> None:
    """
    Internal helper: writes the CSV header row to the output file.

    Creates the parent directory if it does not exist and overwrites
    any existing file content. Called automatically when the file is missing
    or empty.
    """
    # parents=True creates all directories in the path; exist_ok=True suppresses errors
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # "w" mode creates a new file or truncates an existing one
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def log_results(scenario: int, vad_mode: str, results: dict) -> None:
    """
    Appends one row of experiment results to the CSV file.

    If the file does not exist or is empty, the header is written first.
    Each call appends one row — previous results are never overwritten.

    Args:
        scenario (int): Scenario ID.
                        1 = quiet environment
                        2 = noisy environment
                        3 = variable speech rhythm
        vad_mode (str): Label describing the VAD configuration used.
                        Example: "webrtc_agr2", "none", "high"
        results (dict): Metrics dictionary from evaluation/metrics.py.
                        Expected keys:
                          'latency_median', 'latency_p95',
                          'asr_calls_min', 'false_tokens',
                          'wer', 'cer'
    """
    # Write the header if the file does not exist or is empty (size == 0 bytes)
    if not OUT_FILE.exists() or OUT_FILE.stat().st_size == 0:
        _write_header()

    # "a" mode = append — adds the new row without erasing previous content
    with OUT_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),   # Timestamp
            scenario,                                         # Scenario ID
            vad_mode,                                         # VAD configuration label
            # .get() with default 0 prevents KeyError if a metric is missing
            f"{results.get('latency_median', 0):.1f}",      # Median latency
            f"{results.get('latency_p95', 0):.1f}",         # P95 latency
            f"{results.get('asr_calls_min', 0):.2f}",       # Calls per minute
            f"{results.get('false_tokens', 0):.2f}",        # Hallucinations per second
            f"{results.get('wer', 0):.2f}",                 # WER
            f"{results.get('cer', 0):.2f}",                 # CER
        ])