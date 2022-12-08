import torch

model = torch.hub.load(
    "RF5/simple-speaker-embedding", "convgru_embedder", trust_repo=True, device="cpu"
)
