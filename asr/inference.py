import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils.config import MODEL_NAME, SAMPLE_RATE


class ASRModel:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[ASR] Loading model: {model_name}")
        print("[ASR] Downloading from Hugging Face on first run — this may take a while...")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()

        print("[ASR] Model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip()