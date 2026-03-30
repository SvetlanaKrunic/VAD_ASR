# WebRTC VAD + Wav2Vec2 ASR Pipeline


*Batch* and *live* automatic speech recognition (ASR) pipeline using WebRTC Voice Activity Detection (VAD) and Wav2Vec2 acoustic models from Hugging Face.


## Serbian Custom Open Source Dataset

**HuggingFace:** https://huggingface.co/datasets/SvetlanaKrunic/NenadGugl-mudrosti

```
from datasets import load_dataset

ds = load_dataset("SvetlanaKrunic/NenadGugl-mudrosti")
```

## ICIST - paper
S. Krunić, “Parametar Sensitivity and Error Analysis of Voice Activity Detection on English Speech with Cross-Language Evaluation on Custom Serbian Dataset,” in Proc. Int. Conf. on Information Systems and Technologies (ICIST), Serbia, 2026.



## Features

- WebRTC VAD for efficient, low-latency voice activity detection
- Wav2Vec2 ASR via Hugging Face Transformers (auto-downloaded on first run)
- Batch evaluation pipeline with WER, CER, latency, and false token rate metrics
- Live microphone pipeline with real-time transcription
- CSV result logging for experiment comparison
- Supports MP3 and WAV input (auto-converted via ffmpeg)



## Requirements

- Python 3.10+
- ffmpeg installed and available on PATH
```bash
pip install torch torchaudio transformers webrtcvad numpy pyaudio
```

## Data Preparation

Place your audio files in the appropriate `data/raw/scenario_X/` folder.

Reference transcripts go in `data/references/scenario_X.txt` — one sentence per line, matching the alphabetical order of the audio files.

Example `scenario_1.txt`:
```
dobar dan kako ste
ja sam student elektrotehnike
```

The pipeline automatically converts all MP3 and WAV files to 16kHz mono WAV before processing.


## Usage

### Batch Evaluation

Run all scenarios with default VAD aggressiveness:

```bash
python run_pipeline.py
```

Run a specific scenario:

```bash
python run_pipeline.py --scenario 1
```

Run with a specific VAD aggressiveness level:

```bash
python run_pipeline.py --scenario 2 --vad-aggressiveness 3
```

#### Configuration

All parameters are centralized in `utils/config.py`. Edit this file to change model, VAD behavior, or paths.


#### Available arguments:

| Argument | Values | Default | Description |
|---|---|---|---|
| `--scenario` | `1`, `2`, `3` | all | Scenario ID to evaluate |
| `--vad-aggressiveness` | `0`, `1`, `2`, `3` | `2` | WebRTC VAD aggressiveness |

Results are saved to `results/results.csv` automatically.

### Live Transcription

```bash
python live_vad_asr.py
```

Listens via microphone for 60 seconds by default. Press `Ctrl+C` to stop early.



## Output

### Metrics Explained

| Metric | Description |
|---|---|
| **WER** | Word Error Rate — lower is better. Industry target: <10% |
| **CER** | Character Error Rate — finer-grained than WER |
| **Latency median** | Median time from end of speech to ASR result, in ms |
| **Latency P95** | 95th percentile latency — worst case for 95% of utterances |
| **ASR calls/min** | System throughput |
| **False tokens/s** | Hallucinated words per second — lower is better |

### Live vs Batch

| | Live (`live_vad_asr.py`) | Batch (`run_pipeline.py`) |
|---|---|---|
| Input | Microphone | Audio files |
| Output | Terminal only | Terminal + CSV |
| Evaluation | No | WER, CER, latency |
| Use case | Real-time demo | Experiment comparison |

## Model

The default model is [`classla/wav2vec2-xls-r-parlaspeech-hr-lm`](https://huggingface.co/classla/wav2vec2-xls-r-parlaspeech-hr-lm), trained on South Slavic speech (Serbian, Croatian, Bosnian).

To use a different model, change `MODEL_NAME` in `utils/config.py`:

```python
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"  # multilingual
MODEL_NAME = "openai/whisper-small"              # alternative architecture
```

The model is downloaded automatically on first run and cached in `~/.cache/huggingface/`.



## Project Structure

```
vad_asr/
│
├── data/
│   ├── raw/
│   │   ├── scenario_1_quiet/        # original MP3/WAV files, quiet environment
│   │   ├── scenario_2_noisy/        # original MP3/WAV files, noisy environment
│   │   └── scenario_3_variable/     # original MP3/WAV files, variable speech rhythm
│   ├── processed/                   # auto-generated: converted WAV 16kHz mono
│   └── references/
│       ├── scenario_1.txt           # ground-truth transcripts, one per line
│       ├── scenario_2.txt
│       └── scenario_3.txt
│
├── asr/
│   ├── __init__.py
│   └── inference.py                 # Wav2Vec2 model wrapper
│
├── audio/
│   ├── __init__.py
│   ├── mic_stream.py                # microphone input stream (live pipeline)
│   └── file_stream.py               # file input stream (batch pipeline)
│
├── vad/
│   ├── __init__.py
│   ├── base.py                      # abstract VAD base class
│   ├── vad_selector.py              # factory function for VAD selection
│   └── webrtc_vad.py                # WebRTC VAD implementation
│
├── evaluation/
│   ├── __init__.py
│   ├── logger.py                    # CSV result logging
│   └── metrics.py                   # WER, CER, latency aggregation
│
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py               # ffmpeg conversion utilities
│   ├── config.py                    # all system parameters (edit here)
│   └── logger.py                    # Python logging setup
│
├── logs/
│   └── asr_vad.log                  # auto-generated log file
│
├── results/
│   └── results.csv                  # auto-generated evaluation results
│
├── run_pipeline.py                  # batch evaluation entry point
└── live_vad_asr.py                  # live microphone entry point
