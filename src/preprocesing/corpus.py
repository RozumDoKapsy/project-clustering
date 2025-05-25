from pathlib import Path

import pandas as pd

import re
import string

from tqdm import tqdm

from src.preprocesing.text import Lemmatizer, FeatureExtractor, Tokenizer
from src.preprocesing.text import CzechLemmatizer, CVFeatureExtractor, CVTokenizer, JSONStopwordsLoader
from src.preprocesing.models import ModelManager

from typing import List, Union, Optional, Dict


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


class CorpusLemmatizer:
    def __init__(self, lemmatizer: Lemmatizer, feature_extractor: FeatureExtractor, tokenizer: Tokenizer):
        self.lemmatizer = lemmatizer
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.features: List[str] = []
        self.features_lemma_dict: Dict[str, str] = {}

    def process(self, corpus: List[str]):
        self.features = self.feature_extractor.extract_features(corpus)
        self._feature_lemmatize()
        return [self._document_lemmatize(doc) for doc in corpus]

    def _feature_lemmatize(self):
        features_lemmatized = []
        for f in tqdm(self.features):
            features_lemmatized.append(self.lemmatizer.lemmatize(f))
        self.features_lemmatized_dict = dict(zip(self.features, features_lemmatized))
        return self.features_lemmatized_dict

    def _document_lemmatize(self, document: str) -> str:
        tokens = self.tokenizer.tokenize(document)
        tokens_lemmatized = [
            lemma
            for token in tokens
            if self.features_lemmatized_dict.get(token)
            for lemma in self.tokenizer.tokenize(self.features_lemmatized_dict[token])
        ]
        return ' '.join(tokens_lemmatized)


class CorpusLemmatizerBuilder:
    def __init__(self, model_dir: Union[Path, str], model_name: str, stopwords_path: Union[Path, str]):
        self.model_dir = model_dir
        self.model_name = model_name
        self.stopwords_path = stopwords_path

    def build(self) -> CorpusLemmatizer:
        model = ModelManager(self.model_name).load_udpipe_model(self.model_dir)
        stopwords = JSONStopwordsLoader().load(self.stopwords_path)
        cz_lemmatizer = CzechLemmatizer(model)
        cv_extractor = CVFeatureExtractor(stopwords)
        cv_tokenizer = CVTokenizer()
        return CorpusLemmatizer(
            lemmatizer=cz_lemmatizer,
            feature_extractor=cv_extractor,
            tokenizer=cv_tokenizer
        )
