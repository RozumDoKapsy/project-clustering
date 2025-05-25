import os
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

import random

from typing import List, Tuple, Dict, Optional, Union


class ClusterKeywords:
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

    def get_clusters_keywords_df(self) -> pd.DataFrame:
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


class ClusterOAIDefinition:
    def __init__(self, api_key: Optional[str] = None):
        self.client = self._get_openai_client(api_key)
        self.clusters_definition = {}

    @staticmethod
    def _get_openai_client(api_key: Optional[str] = None):
        if api_key:
            return OpenAI(api_key=api_key)
        else:
            from dotenv import load_dotenv
            load_dotenv()
            return OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def get_clusters_definition(
            self
            , clusters_keywords: Dict[any, List[str]]
            , labels: np.ndarray
            , documents: Union[pd.Series, List[str]]
            ) -> Dict[any, Dict[str, str]]:

        clusters_definition = {}
        for cluster, kw in clusters_keywords.items():
            idxs = np.where(labels == cluster)[0]
            examples = self._get_examples(documents[idxs])
            response = self._get_definition(kw, examples)
            clusters_definition[cluster] = self._parse_response(response)
        self.clusters_definition = clusters_definition
        return self.clusters_definition

    def get_clusters_definition_df(self) -> pd.DataFrame:
        if self.clusters_definition is None:
            raise ValueError('Cluster definition dict is empty. Extract cluster definition first.')

        df = pd.DataFrame.from_dict(self.clusters_definition, orient='index').reset_index()
        df.columns = ['cluster_id', 'name', 'definition']
        return df

    def _get_definition(self, keywords: List[str], examples: List[str]):
        prompt = f'''
        Vytvoř krátký název a definici (2-3 věty) klastrů výzkumných projektů na základě násldujcích klíčových slov 
        a příkladů projektů.
        Klíčová slova: {' '.join(keywords)}
        Příklady projetků: {'\n'.join(examples)}
        
        Vrať výstup v následujícím JSON formátu.
        {{"name": ".....", "definition": "....."}}
        '''

        response = self.client.responses.create(
            model='gpt-4o-mini',
            input=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': prompt}
                    ]
                }
            ],
            temperature=0.2
        )

        return response

    @staticmethod
    def _get_examples(documents: Union[pd.Series, List[str]], n_examples: int = 3) -> List[str]:
        if isinstance(documents, pd.Series):
            documents = documents.tolist()
        return random.sample(documents, n_examples)

    @staticmethod
    def _parse_response(response) -> Dict[str, str]:
        raw_text = response.output[0].content[0].text

        if raw_text.startswith("```json"):
            raw_text = raw_text.strip("``` \n")
            raw_text = raw_text.replace("```", "").replace("json", "").strip()

        return json.loads(raw_text)
