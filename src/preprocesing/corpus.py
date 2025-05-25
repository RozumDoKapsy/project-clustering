from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from corpy.udpipe import Model as UDPipeModel

import re
import string

from tqdm import tqdm

from typing import List, Union, Optional


def load_stopwords(file_path: Union[Path, str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    file_path = Path(file_path)
    if file_path.exists():
        stopwords_df = pd.read_json(file_path)
        stopwords = stopwords_df[stopwords_df.columns[0]].tolist()
    else:
        raise FileNotFoundError(f'Stopwords file not found at location: from {file_path}')

    if custom_stopwords:
        stopwords.extend(custom_stopwords)
    return stopwords


class CorpusBuilder:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data
        self.corpus = pd.DataFrame()

    @staticmethod
    def _load_from_csv(file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f'CSV file not found at path {file_path}.')
        return pd.read_csv(file_path)

    @staticmethod
    def _load_from_excel(file_path: Path) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f'Excel file not found at path {file_path}.')
        return pd.read_excel(file_path)

    def load_data_from_file(self, file_path: Union[Path, str]) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            self.data = self._load_from_csv(file_path)
            return self.data
        elif file_path.suffix in ['.xls', '.xlsx']:
            self.data = self._load_from_excel(file_path)
            return self.data
        else:
            raise ValueError(f'Unsupported file type: {file_path.suffix}')

    def create_corpus(self, id_col: str, text_cols: List[str], join_char: str) -> pd.DataFrame:
        if self.data is None or self.data.empty:
            raise ValueError('No data loaded. Either load data directly or from file.')

        corpus = self.data[[id_col]].copy()
        corpus['corpus'] = self.data[text_cols].astype(str).agg(join_char.join, axis=1)
        corpus = corpus.dropna()
        self.corpus = corpus
        return self.corpus

    def save(self, file_path: Union[Path, str]):
        file_path = Path(file_path)
        if self.corpus is None or self.corpus.empty:
            raise ValueError('Corpus is empty')

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == '.csv':
            self.corpus.to_csv(file_path, index=False)
        elif file_path.suffix == '.xlsx':
            self.corpus.to_excel(file_path, index=False)
        else:
            raise ValueError(f'Unsupported file type: {file_path.suffix}.')


class CorpusCleaner:
    def __init__(self):
        pass

    def clean(self, corpus: List[str]):
        return [self._document_clean(doc) for doc in corpus]

    @staticmethod
    def _document_clean(document: str) -> str:
        document = document.lower()
        document = re.sub(r'\d+', '', document)
        document = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document)
        document = re.sub(r'\s+', ' ', document)
        return document


class CorpusCzechLemmatizer:
    def __init__(self, model: UDPipeModel):
        self.model = model
        self.features = np.array([], dtype=str)
        self.features_lemmatized_dict = {}

    def lemmatize(self, corpus: List[str], stopwords: List[str]):
        self._get_corpus_features(corpus, stopwords)
        self._feature_lemmatize()
        return [self._document_lemmatize(doc) for doc in corpus]

    def _get_corpus_features(self, corpus: List[str], stopwords: List[str]):
        cv = CountVectorizer(min_df=1, stop_words=stopwords)
        cv.fit_transform(corpus)
        self.features = cv.get_feature_names_out()
        return self.features

    def _text_lemmatize(self, text: str) -> str:
        sentence = []
        for s in self.model.process(text):
            for w in s.words:
                if '<root>' not in w.lemma:
                    sentence.append(w.lemma)
        return ' '.join(sentence)

    def _feature_lemmatize(self):
        features_lemmatized = []
        for f in tqdm(self.features):
            features_lemmatized.append(self._text_lemmatize(f))
        self.features_lemmatized_dict = dict(zip(self.features, features_lemmatized))
        return self.features_lemmatized_dict

    def _document_lemmatize(self, document: str) -> str:
        tokens = document.split(' ')
        tokens_lemmatized = [
            lemma
            for token in tokens
            if self.features_lemmatized_dict.get(token)
            for lemma in self.features_lemmatized_dict[token].split(' ')
        ]
        return ' '.join(tokens_lemmatized)