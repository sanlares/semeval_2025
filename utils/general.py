import torch
from sentence_transformers import util

def custom_similarity(a: torch.Tensor, b: torch.Tensor, device: str) -> torch.Tensor:
    """Custom similarity function optimized for fact-checking retrieval."""
    a = a.to(device)
    b = b.to(device)
    sim = util.cos_sim(a, b).float()
    return sim 