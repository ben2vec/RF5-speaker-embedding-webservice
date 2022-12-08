WHISPER_PORT=9000
EMBED_PORT=9001

WHISPER_DOCKERHUB=thoppe/openai-whisper-asr-webservice-predownload-medium:12-08-2022
EMBED_DOCKERHUB=thoppe/rf5_speaker_embedding_webservice

run_whisper:
	docker run -p $(WHISPER_PORT):9000 -e ASR_MODEL=medium $(WHISPER_DOCKERHUB)

pull_whisper:
	docker pull $(WHISPER_DOCKERHUB)

build_embed:
	docker build -t $(EMBED_DOCKERHUB) .

run_embed:
	docker run -p $(EMBED_PORT):9000 $(EMBED_DOCKERHUB)

push_embed:
	docker push $(EMBED_DOCKERHUB)
