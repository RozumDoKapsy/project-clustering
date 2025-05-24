import numpy as np
import pandas as pd
from sklearn.metrics.cluster import silhouette_score, silhouette_samples


class ClusterEvaluator:
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels

    def evaluate(self, metric: str):
        return {
            'cluster_count': self._cluster_count(),
            'outlier_ratio': self._outlier_ratio(),
            'silhouette_score': self._silhouette_score(metric, include_outlier=False),
            'silhouette_score_outliers': self._silhouette_score(metric, include_outlier=True),
            'gini_coefficient': self._gini_coefficient(),
            'clusters_silhouette_scores': self._cluster_silhouette_score(metric)
        }

    def _silhouette_score(self, metric: str, include_outlier: False) -> float:
        if include_outlier:
            return silhouette_score(self.embeddings, self.labels, metric=metric)
        else:
            valid_labels_indices = np.where(self.labels != -1)[0]
            valid_labels = self.labels[valid_labels_indices]
            valid_embeddings = self.embeddings[valid_labels_indices]
            return silhouette_score(valid_embeddings, valid_labels, metric=metric)

    def _cluster_count(self) -> int:
        return pd.Series(self.labels[self.labels >= 0]).nunique()

    def _outlier_ratio(self) -> float:
        num_outliers = np.sum(self.labels == -1)
        return float(num_outliers / len(self.labels))

    def _gini_coefficient(self) -> float:
        cluster_counts = pd.Series(self.labels).value_counts()
        array = np.sort(np.array(cluster_counts))
        n = array.shape[0]
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * array) - (n + 1) * np.sum(array)) / (n * np.sum(array))

    def _cluster_silhouette_score(self, metric: str) -> dict:
        sample_silhouette_values = silhouette_samples(self.embeddings, self.labels, metric=metric)
        unique_clusters = np.unique(self.labels)
        cluster_silhouette_scores = {}
        for cluster in unique_clusters:
            cluster_score = sample_silhouette_values[self.labels == cluster]
            cluster_silhouette_scores[int(cluster)] = float(np.mean(cluster_score))
        return cluster_silhouette_scores
