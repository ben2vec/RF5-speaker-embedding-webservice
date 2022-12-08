# Dockerized webservice for Simple speaker embeddings

Source repo: https://github.com/RF5/simple-speaker-embedding

## Getting started

Start the docker containers, one for OpenAI's [Whisper](https://github.com/ahmetoner/whisper-asr-webservice) model and one for RF5.

```
docker run -p -d 9000:9000 -e ASR_MODEL=medium thoppe/openai-whisper-asr-webservice-predownload-medium:12-08-2022
docker run -p -d 9001:9000 thoppe/rf5_speaker_embedding_webservice
```

### Sample code [test_api.py](test_api.py)
```python
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

vecs = r.json()
sim = np.round(cosine_similarity(vecs), 2)
print(sim)
```

### Sample output

```
 I want Iguodala. That's right. And I know what that sounds like. And it's not fair to Steph because so much defensive attention is paid to him. And I'm not saying Steph's not a better shooter. He's a way better shooter. Iguodala's got ice water in his veins. My money is on number 30.
[[1.   0.63 0.62 0.43 0.6  0.61 0.24]
 [0.63 1.   0.67 0.73 0.61 0.56 0.05]
 [0.62 0.67 1.   0.46 0.57 0.7  0.11]
 [0.43 0.73 0.46 1.   0.52 0.38 0.16]
 [0.6  0.61 0.57 0.52 1.   0.5  0.17]
 [0.61 0.56 0.7  0.38 0.5  1.   0.06]
 [0.24 0.05 0.11 0.16 0.17 0.06 1.  ]]
```

### References

Sample video source
+ [Andre Iguodala should take the last shot, not Steph Curry - Max Kellerman | First Take](https://www.youtube.com/watch?v=mLFZOTb4n_s)

Associated images on DockerHub

+   https://hub.docker.com/r/thoppe/openai-whisper-asr-webservice-predownload-medium
+   https://hub.docker.com/r/thoppe/rf5_speaker_embedding_webservice

