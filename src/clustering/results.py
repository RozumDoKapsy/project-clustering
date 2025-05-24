from pathlib import Path

import numpy as np
import pandas as pd

from typing import List, Union


class ClusterResultSaver:
    def __init__(self, labels: np.ndarray):
        self.labels = labels
        self.clustering_results = pd.DataFrame()

    def assign_clusters(self, ids: List[Union[str, int]], documents: List[str]) -> pd.DataFrame:
        labels = self.labels
        if len(labels) != len(ids) or len(labels) != len(documents):
            raise ValueError(f'Provided data are not of the same length: '
                             f'labels-{len(labels)}, ids-{len(ids)}, documents-{len(documents)}.')

        self.clustering_results = pd.DataFrame({
            "cluster_id": self.labels,
            "document_id": ids,
            "document_text": documents
        })
        return self.clustering_results

    def save(self, file_path: Union[Path, str]):
        file_path = Path(file_path)
        if self.clustering_results is None or self.clustering_results.empty:
            raise ValueError('Corpus is empty')

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == '.csv':
            self.clustering_results.to_csv(file_path, index=False)
        elif file_path.suffix == '.xlsx':
            self.clustering_results.to_excel(file_path, index=False)
        else:
            raise ValueError(f'Unsupported file type: {file_path.suffix}.')
