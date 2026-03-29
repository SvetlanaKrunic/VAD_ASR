SAMPLE_RATE = 16000 # Wav2Vec2, WebRTC VAD - 16000 Hz.
CHUNK_DURATION_MS = 30 # WebRTC VAD: 10ms, 20ms ili 30ms 
FRAME_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) # in example: 16000 * 30 / 1000 = 480 samples in chunk

#VAD
# 0 – lets almost everything through, good for quiet rooms without noise
# 1 – slightly stricter, suitable for office environments
# 2 – balance between sensitivity and noise robustness (recommended)
# 3 – aggressively cuts background noise, but may clip the ends of words
VAD_AGGRESSIVENESS = 2
VAD_PADDING_MS = 600
VAD_TRIGGER_LEVEL = 0.8

# ASR
MIN_SPEECH_DURATION_SEC = 0.5
MAX_UTTERANCE_SEC = 25

# MODEL, DATA and RESULTS

# Model on Hugging Face Hub
# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
MODEL_NAME = "classla/wav2vec2-xls-r-parlaspeech-hr-lm"

#   data/raw/           – original MP3/WAV files
#   data/processed/     – converted to WAV 16kHz mono
#   data/references/    – .txt transcriptions
DATA_DIR = "data"

RESULTS_DIR = "results"