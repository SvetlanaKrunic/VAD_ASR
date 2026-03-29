# =============================================================================
# evaluation/metrics.py – WER, CER, and evaluation result aggregation
# =============================================================================
# WER (Word Error Rate) and CER (Character Error Rate) are the standard metrics
# for evaluating ASR systems. Both are computed using the Levenshtein edit
# distance algorithm, which finds the minimum number of operations
# (substitutions, deletions, insertions) needed to transform the hypothesis
# into the reference transcript.
# =============================================================================

import numpy as np
from pathlib import Path


def _levenshtein_ops(ref, hyp):
    """
    Computes the edit distance between a reference and a hypothesis sequence
    and returns the counts of substitutions, deletions, and insertions.

    Uses dynamic programming (DP table) — the classic Levenshtein algorithm.
    Time complexity:  O(|ref| * |hyp|)
    Space complexity: O(|ref| * |hyp|)

    Args:
        ref (list): List of reference tokens (words or characters).
        hyp (list): List of hypothesis tokens (ASR output).

    Returns:
        tuple: (distance, substitutions, deletions, insertions)
    """
    # Initialize the DP table with dimensions (|ref|+1) x (|hyp|+1)
    # d[i][j] = edit distance between ref[:i] and hyp[:j]
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)

    # backtrace[i][j] stores which operation was used to reach d[i][j]
    # Values: -1 = match, 0 = substitution, 1 = deletion, 2 = insertion
    backtrace = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)

    # Initialize the first column: transforming ref[:i] into an empty sequence
    # requires i deletions
    for i in range(len(ref) + 1):
        d[i][0] = i

    # Initialize the first row: transforming an empty sequence into hyp[:j]
    # requires j insertions
    for j in range(len(hyp) + 1):
        d[0][j] = j

    # Fill the DP table from top-left to bottom-right
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Tokens match — no cost; inherit value from the diagonal cell
                d[i][j] = d[i - 1][j - 1]
                backtrace[i][j] = -1  # mark as match
            else:
                # Compute the cost of each possible operation and take the minimum
                del_cost = d[i - 1][j] + 1      # deletion: remove token from ref
                ins_cost = d[i][j - 1] + 1      # insertion: add token to hyp
                sub_cost = d[i - 1][j - 1] + 1  # substitution: replace token

                d[i][j] = min(del_cost, ins_cost, sub_cost)

                # Record which operation was selected for backtracking
                if d[i][j] == del_cost:
                    backtrace[i][j] = 1   # deletion
                elif d[i][j] == ins_cost:
                    backtrace[i][j] = 2   # insertion
                else:
                    backtrace[i][j] = 0   # substitution

    # Backtrack from d[|ref|][|hyp|] to d[0][0] to count each operation type
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0

    while i > 0 or j > 0:
        op = backtrace[i][j]

        if op == -1:   # match — both pointers move diagonally
            i -= 1
            j -= 1
        elif op == 0:  # substitution — both pointers move diagonally, error counted
            subs += 1
            i -= 1
            j -= 1
        elif op == 1:  # deletion — only the ref pointer moves up
            dels += 1
            i -= 1
        elif op == 2:  # insertion — only the hyp pointer moves left
            ins += 1
            j -= 1
        else:
            break

        # Clamp indices — they must never go below zero
        if i < 0:
            i = 0
        if j < 0:
            j = 0

        # Stop backtracking when both pointers reach (0, 0)
        if i == 0 and j == 0:
            break

    return d[len(ref)][len(hyp)], subs, dels, ins


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    """
    Computes Word Error Rate (WER) as a percentage.

    WER = (S + D + I) / N * 100
    where N = total words in the reference, S = substitutions,
    D = deletions, I = insertions.

    A WER of 0% means perfect transcription. The industry target for a
    good ASR system is typically below 10%.

    Args:
        hypotheses (list[str]): List of ASR outputs (predicted transcripts).
        references (list[str]): List of ground-truth transcripts.

    Returns:
        float: WER in percent.
    """
    total_words = 0
    total_errs = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()   # split on whitespace to get word list
        ref_words = ref.split()
        dist, _, _, _ = _levenshtein_ops(ref_words, hyp_words)
        total_errs += dist
        total_words += len(ref_words)

    return 100.0 * total_errs / total_words if total_words > 0 else 0.0


def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """
    Computes Character Error Rate (CER) as a percentage.

    Same as WER but operates at the character level — useful for languages
    with agglutination or when the model makes letter-level errors.

    Args:
        hypotheses (list[str]): List of ASR outputs.
        references (list[str]): List of ground-truth transcripts.

    Returns:
        float: CER in percent.
    """
    total_chars = 0
    total_errs = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_chars = list(hyp)   # list() splits the string into individual characters
        ref_chars = list(ref)
        dist, _, _, _ = _levenshtein_ops(ref_chars, hyp_chars)
        total_errs += dist
        total_chars += len(ref_chars)

    return 100.0 * total_errs / total_chars if total_chars > 0 else 0.0


def load_references(file_path: str) -> list[str]:
    """
    Loads ground-truth transcripts from a plain text file.

    Expected format: one sentence per line, UTF-8 encoding.
    Empty lines are automatically skipped.

    Args:
        file_path (str): Path to the .txt file containing reference transcripts.

    Returns:
        list[str]: List of reference transcripts, one per line.
    """
    return [
        line.strip()
        for line in Path(file_path).read_text(encoding="utf-8").splitlines()
        if line.strip()   # skip blank lines
    ]


def aggregate_results(results: list[dict], references: list[str] | None = None) -> dict:
    """
    Aggregates all metrics from a single experiment run into one dictionary.

    Combines latency metrics, ASR throughput, and transcription accuracy
    into a compact summary suitable for CSV logging.

    Args:
        results (list[dict]): List of per-utterance result dictionaries. Each must contain:
                              - "text" (str): ASR transcript
                              - "duration" (float): audio segment duration in seconds
                              - "latency" (float): time from end of speech to ASR result, in ms
                              - "start_time" / "end_time" (float): Unix wall-clock timestamps
        references (list[str] | None): Ground-truth transcripts for WER/CER computation.
                                       If None, WER and CER remain 0.

    Returns:
        dict: Aggregated metrics:
              - "wer": Word Error Rate in %
              - "cer": Character Error Rate in %
              - "false_tokens": hallucinated insertions per second
              - "latency_median": median ASR latency in ms
              - "latency_p95": 95th percentile latency in ms (worst case for 95% of utterances)
              - "asr_calls_min": number of ASR calls per minute (throughput)
    """
    if not results:
        return {}

    latencies = [r["latency"] for r in results]
    transcripts = [r["text"] for r in results]

    # Total duration of all audio segments in seconds
    total_duration = sum(r["duration"] for r in results)

    wer = cer = 0.0
    false_tokens_per_sec = 0.0

    if references is not None:
        total_insertions = 0
        total_words = 0
        total_errs = 0

        for hyp, ref in zip(transcripts, references):
            hyp_words = hyp.split()
            ref_words = ref.split()
            dist, subs, dels, ins = _levenshtein_ops(ref_words, hyp_words)
            total_errs += dist
            total_words += len(ref_words)
            # Insertions are "invented" words — the model is hallucinating them
            total_insertions += ins

        wer = 100.0 * total_errs / total_words if total_words > 0 else 0.0

        # False tokens per second: how often the model hallucinates a word
        # Lower is better — 0.0 means no hallucinations
        false_tokens_per_sec = total_insertions / total_duration if total_duration > 0 else 0.0
        cer = compute_cer(transcripts, references)

    return {
        "wer": wer,
        "cer": cer,
        "false_tokens": false_tokens_per_sec,
        # np.median is robust to outliers unlike the arithmetic mean
        "latency_median": float(np.median(latencies)),
        # 95th percentile: 95% of utterances have latency below this value
        "latency_p95": float(np.percentile(latencies, 95)),
        # Throughput: how many utterances are processed per minute
        "asr_calls_min": len(results) / ((results[-1]["end_time"] - results[0]["start_time"]) / 60),
    }