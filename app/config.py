import os
import torch

class Settings:
    PROJECT_NAME: str = "German Real-Time TTS API"
    MODEL_NAME: str = "tts_models/de/thorsten/vits"
    DEVICE: str = "cpu"
    
    # Optimize specifically for the Ryzen 5 1600's 12 threads
    NUM_THREADS: int = 12

def configure_system():
    torch.set_num_threads(Settings.NUM_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(Settings.NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(Settings.NUM_THREADS)
    # Prevents fragmentation of memory on system RAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"