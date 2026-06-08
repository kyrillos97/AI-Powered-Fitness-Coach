import os
import torch
import torchaudio

class VoiceIdentityManager:
    """
    Manages voice identity (speaker verification) using SpeechBrain's ECAPA-TDNN model.
    
    SpeechBrain is imported lazily (inside methods) to prevent the k2_fsa lazy module
    error from crashing the FastAPI server at startup.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.classifier = None
        self.enrolled_embedding = None
        self.similarity_threshold = 0.50
        self.enabled = False  # Will be set True once model loads
        self.needs_enrollment = True  # True until a voice is enrolled
        self._load_model()

    def _load_model(self):
        """Lazy-load SpeechBrain so a bad optional sub-module (k2_fsa etc.) 
        does NOT propagate an error at server startup."""
        try:
            # Import ONLY here, not at module level
            from speechbrain.inference.speaker import EncoderClassifier
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxcelex",
                savedir="pretrained_models/spkrec-ecapa-voxcelex",
                run_opts={"device": "cpu"}
            )
            self.enabled = True
            print("[VoiceID] Speaker verification model loaded successfully.")
        except Exception as e:
            self.enabled = False
            print(f"[VoiceID] Warning: Could not load speaker model — voice verification disabled. ({e})")

    def _load_audio(self, wav_path: str):
        """Load a wav file and resample to 16 kHz for ECAPA."""
        signal, fs = torchaudio.load(wav_path)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def enroll_voice(self, wav_path: str) -> bool:
        """Extract and store the speaker embedding from the audio file."""
        if not self.enabled:
            print("[VoiceID] Enroll skipped — model not loaded.")
            return False
        try:
            signal = self._load_audio(wav_path)
            with torch.no_grad():
                self.enrolled_embedding = self.classifier.encode_batch(signal)
            self.needs_enrollment = False
            print("[VoiceID] Voice enrolled successfully.")
            return True
        except Exception as e:
            print(f"[VoiceID] Enroll error: {e}")
            return False

    def verify_voice(self, wav_path: str) -> bool:
        """
        Verify whether the audio matches the enrolled speaker.
        Returns True (allow) if:
          - The model is disabled, OR
          - No voice has been enrolled yet, OR
          - The cosine similarity exceeds the threshold.
        """
        if not self.enabled:
            return True   # bypass — model not loaded
        if self.enrolled_embedding is None:
            return True   # bypass — no enrollment yet
        try:
            signal = self._load_audio(wav_path)
            with torch.no_grad():
                cur_emb = self.classifier.encode_batch(signal)
            score = torch.nn.functional.cosine_similarity(
                self.enrolled_embedding.squeeze(1),
                cur_emb.squeeze(1)
            ).item()
            match = score > self.similarity_threshold
            print(f"[VoiceID] Score={score:.3f} threshold={self.similarity_threshold} match={match}")
            return match
        except Exception as e:
            print(f"[VoiceID] Verify error: {e}")
            return True   # fail-open so the user is never locked out by a bad audio clip
