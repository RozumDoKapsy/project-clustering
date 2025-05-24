from pathlib import Path
import numpy as np

from src.clustering.models import BaseClusteringModel
from src.clustering.evaluation import ClusterEvaluator
from src.clustering.results import ClusterResultSaver

from typing import List, Union, Optional


class ClusteringPipeline:
    def __init__(self, model: BaseClusteringModel):
        self.model = model

    def run(
            self,
            embeddings: np.ndarray,
            ids: List[Union[str, int]],
            documents: List[str],
            save: bool = False,
            file_path: Optional[Union[Path, str]] = None,
    ) -> BaseClusteringModel:
        self.model.fit(embeddings)
        evaluator = ClusterEvaluator(embeddings, labels=self.model.labels)
        self.model.evaluation_results = evaluator.evaluate(self.model.params['metric'])

        if save:
            saver = ClusterResultSaver(self.model.labels)
            saver.assign_clusters(ids, documents)
            saver.save(file_path)
        return self.model
