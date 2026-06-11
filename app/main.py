from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import configure_system, Settings
from app.tts_engine import TTSEngine

# Apply CPU optimizations immediately at startup
configure_system()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the heavy 330MB VITS model once into RAM
    TTSEngine.load_model()
    yield
    # Shutdown: Clear memory hooks smoothly
    TTSEngine.unload_model()

app = FastAPI(title=Settings.PROJECT_NAME, lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="The German text string to convert to speech.",
        examples=["Guten Tag! Die API läuft einwandfrei auf Ihrem Prozessor."]
    )

def cleanup_temp_file(file_path: str):
    """Deletes the temporary generated WAV file after it has been streamed out."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

@app.post("/api/v1/tts", response_class=FileResponse)
def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Synthesizes German text into speech.
    Returns an audio/wav stream and cleans up the server storage automatically.
    """
    try:
        # FastAPI handles regular 'def' endpoints inside an internal threadpool,
        # keeping the main event loop unblocked for other requests.
        wav_path = TTSEngine.synthesize(text=request.text)
        
        # Register the file cleanup to run immediately after the file finishes streaming
        background_tasks.add_task(cleanup_temp_file, wav_path)
        
        return FileResponse(
            path=wav_path, 
            media_type="audio/wav", 
            filename="speech.wav"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Engine failure: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Spin up Uvicorn deployment worker
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)