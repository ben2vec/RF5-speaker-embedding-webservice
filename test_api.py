import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

WHISPER_PORT = 9000
EMBED_PORT = 9001

whisper_url = f"http://localhost:{WHISPER_PORT}/asr"
embed_url = f"http://localhost:{EMBED_PORT}/embed"

f_audio = "AndreIguodala_ESPN.mp3"

with open(f_audio, "rb") as FIN:
    raw_audio = FIN.read()

r = requests.post(whisper_url, files={"audio_file": raw_audio})
assert r.ok

js = r.json()
print(js["text"])

r = requests.post(embed_url, files={"audio_file": raw_audio, "whisper_file": r.content})

assert r.ok

vecs = np.array(r.json())
print(vecs)
sim = cosine_similarity(vecs)
print(sim)

sim = euclidean_distances(vecs)
print(sim)
