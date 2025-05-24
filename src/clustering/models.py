import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering


class BaseClusteringModel:
    def __init__(self, params: dict):
        self.params = params
        self.labels = None
        self.evaluation_results = {}

    def fit(self, embeddings: np.ndarray):
        raise NotImplementedError()


class HDBSCANClusteringModel(BaseClusteringModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.model = HDBSCAN(**params)

    def fit(self, embeddings: np.ndarray):
        self.model.fit(embeddings)
        self.labels = self.model.labels_


class AgglomerativeClusteringModel(BaseClusteringModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.model = AgglomerativeClustering(**params)

    def fit(self, embeddings: np.ndarray):
        self.model.fit(embeddings)
        self.labels = self.model.labels_
