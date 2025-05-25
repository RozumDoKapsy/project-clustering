import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List, Tuple, Dict


class ClusterDefinition:
    def __init__(self):
        self.tfidf = None
        self.tfidf_matrix = np.ndarray([])
        self.cluster_kw = {}

    def _create_tfidf(self, documents: List[str], **kwargs) -> Tuple[TfidfVectorizer, np.ndarray]:
        self.tfidf = TfidfVectorizer(**kwargs)
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        return self.tfidf, self.tfidf_matrix

    def fit(self, documents: List[str], **kwargs):
        self._create_tfidf(documents, **kwargs)

    def _extract_cluster_keywords(self, labels: np.ndarray, cluster: int, topn: int) -> List[str]:
        idxs = np.where(labels == cluster)[0]
        cluster_tfidf = self.tfidf_matrix[idxs].toarray().mean(axis=0)
        sorted_cluster_tfidf = np.argsort(cluster_tfidf)[::-1]
        return self._extract_topn_keywords(cluster_tfidf, sorted_cluster_tfidf, topn)

    def _extract_topn_keywords(self, cluster_tfidf: np.ndarray
                               , sorted_cluster_tfidf: np.ndarray, topn: int) -> List[str]:
        feature_idxs = [idx for idx in sorted_cluster_tfidf if cluster_tfidf[idx] > 0]
        features = list(self.tfidf.get_feature_names_out()[feature_idxs[:topn]])
        return features

    def extract_cluster_keywords(self, labels: np.ndarray, topn: int) -> Dict[any, List[str]]:
        unique_clusters = pd.Series(labels[labels >= 0]).unique()
        cluster_kw = {}
        for cluster in unique_clusters:
            cluster_kw[cluster] = self._extract_cluster_keywords(labels, cluster, topn)
        self.cluster_kw = cluster_kw
        return self.cluster_kw

    def get_clusters_definition_df(self) -> pd.DataFrame:
        if self.cluster_kw is None:
            raise ValueError('Cluster keywords dict is empty. Extract cluster keywords first.')

        cluster_ids = []
        keywords = []
        for key, value in self.cluster_kw.items():
            cluster_ids.append(key)
            keywords.append(', '.join(value))

        return pd.DataFrame({
            'cluster_id': cluster_ids,
            'keywords': keywords
        })


