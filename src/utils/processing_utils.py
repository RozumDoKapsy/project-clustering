import numpy as np
import umap


def umap_reduce(embeddings: np.ndarray, params: dict) -> np.ndarray:
    umap_model = umap.UMAP(**params, random_state=42)
    return umap_model.fit_transform(embeddings)