from abc import ABC, abstractmethod
from pathlib import Path
from corpy.udpipe import Model as UDPipeModel

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from typing import List, Optional, Union


class Lemmatizer(ABC):
    @abstractmethod
    def lemmatize(self, text: str) -> str:
        pass


class CzechLemmatizer(Lemmatizer):
    def __init__(self, model: UDPipeModel):
        self.model = model

    def lemmatize(self, text: str) -> str:
        sentence = []
        for s in self.model.process(text):
            for w in s.words:
                if '<root>' not in w.lemma:
                    sentence.append(w.lemma)
        return ' '.join(sentence)


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, documents: List[str]) -> np.ndarray:
        pass


class CVFeatureExtractor(FeatureExtractor):
    def __init__(self, stopwords: Optional[List[str]] = None):
        self.cv = CountVectorizer(min_df=1, stop_words=stopwords)

    def extract_features(self, documents: List[str]) -> np.ndarray:
        self.cv.fit_transform(documents)
        return self.cv.get_feature_names_out()


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class CVTokenizer(Tokenizer):
    def __init__(self):
        self.cv = CountVectorizer()

    def tokenize(self, text: str) -> List[str]:
        return self.cv.build_tokenizer()(text)


class StopwordsLoader(ABC):
    @abstractmethod
    def load(self, file_path: Union[Path, str]) -> List[str]:
        pass

    @abstractmethod
    def add_custom_stopwords(self, custom_stopwords: List[str]):
        pass


class JSONStopwordsLoader(StopwordsLoader):
    def __init__(self):
        self.stopwords: List[str] = []

    def load(self, file_path: Union[Path, str]) -> List[str]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'Stopwords file not found at location: from {file_path}')
        stopwords_df = pd.read_json(file_path)
        self.stopwords = stopwords_df[stopwords_df.columns[0]].tolist()
        return self.stopwords

    def add_custom_stopwords(self, custom_stopwords: List[str]) -> None:
        self.stopwords.extend(custom_stopwords)
