import os
import tempfile
from typing import Optional
from TTS.api import TTS
from app.config import Settings

class TTSEngine:
    _instance: Optional[TTS] = None

    @classmethod
    def load_model(cls) -> None:
        """Loads the model into memory exactly once during server startup."""
        if cls._instance is None:
            print(f"Loading {Settings.MODEL_NAME} onto {Settings.DEVICE}...")
            # Automatically downloads if not cached locally
            cls._instance = TTS(model_name=Settings.MODEL_NAME, progress_bar=False)
            cls._instance.to(Settings.DEVICE)
            print("Model loaded successfully.")

    @classmethod
    def unload_model(cls) -> None:
        """Cleans up memory when the server shuts down."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            print("Model unloaded from memory.")

    @classmethod
    def synthesize(cls, text: str) -> str:
        """
        Generates a WAV file from text.
        Returns the absolute file path to the temporary audio file.
        """
        if cls._instance is None:
            raise RuntimeError("TTS Model is not loaded. Call load_model() first.")
        
        # Create a secure, unique temp file that won't collision under high traffic
        fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Close file descriptor so Coqui TTS can open/write to it safely
        
        try:
            cls._instance.tts_to_file(text=text, file_path=temp_file_path)
            return temp_file_path
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e