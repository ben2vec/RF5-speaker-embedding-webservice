from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import ffmpeg
import numpy as np
from threading import Lock
import torch
from typing import BinaryIO

from tqdm import tqdm
import json
import pandas as pd


SAMPLE_RATE = 16000

app = FastAPI(
    title="RF5-speaker-embedding-webservice",
    description="Webservice to access https://github.com/RF5/simple-speaker-embedding",
    # version=projectMetadata['Version'],
    # contact={
    #    "url": projectMetadata['Home-page']
    # },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        # "url": projectMetadata['License']
    },
)

model = torch.hub.load("RF5/simple-speaker-embedding", "convgru_embedder", device="cpu")
model.eval()
model_lock = Lock()

# model_name= os.getenv("ASR_MODEL", "base")
# if torch.cuda.is_available():
#    model = whisper.load_model(model_name).cuda()
# else:
#    model = whisper.load_model(model_name)
# model_lock = Lock()


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/embed", tags=["Endpoints"])
def transcribe(
    audio_file: UploadFile = File(...),
    whisper_file: UploadFile = File(...),
):

    js = json.load(whisper_file.file)
    df = pd.DataFrame(js["segments"]).set_index("id")
    audio = load_audio(audio_file.file)
    wav = torch.from_numpy(audio).float()
    
    X = []
    for idx, row in tqdm(df.iterrows()):
        sr0 = int(SAMPLE_RATE*row.start)
        sr1 = int(SAMPLE_RATE*row.end)
        w1 = audio[sr0:sr1]
        
        with model_lock:
            embedding = model(wav[None]).detach().cpu().numpy().ravel()

        X.append(embedding.tolist())

    return X

def load_audio(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0




"""
@app.post("/detect-language", tags=["Endpoints"])
def language_detection(
                audio_file: UploadFile = File(...),
                ):

    # load audio and pad/trim it to fit 30 seconds
    audio = load_audio(audio_file.file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)
    
    result = { "detected_language": LANGUAGES[detected_lang_code],
              "langauge_code" : detected_lang_code }

    return result


def run_asr(file: BinaryIO, task: Union[str, None], language: Union[str, None] ):
    audio = load_audio(file)
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language    
    with model_lock:   
        result = model.transcribe(audio, **options_dict)
        
    return result

"""
