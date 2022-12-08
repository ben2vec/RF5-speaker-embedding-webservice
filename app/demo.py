import requests
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

name = "AndreIguodala_ESPN"

src_dir = Path("..")
f_whisper = src_dir / Path(name).with_suffix(".json")
assert f_whisper.exists()

f_audio = src_dir / Path(name).with_suffix(".mp3")
assert f_audio.exists()
url = "http://localhost:8000/embed"

with open(f_whisper, "rb") as FIN:
    raw_whisper = FIN.read()

with open(f_audio, "rb") as FIN:
    raw_audio = FIN.read()

r = requests.post(url, files={"audio_file": raw_audio, "whisper_file": raw_whisper})

assert r.ok
X = r.json()
print(np.round(cosine_similarity(X), 2))
