from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
import numpy as np

from nltk import tokenize
import nltk

from tqdm import tqdm

from typing import List, Union, Optional, Tuple

from src.preprocesing.models import ModelManager
from src.utils.processing_utils import umap_reduce

nltk.download('punkt_tab')


class EmbeddingGenerator:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def create_document_embeddings(self, text: str, sent_tokenize: bool = True) -> np.ndarray:
        if sent_tokenize:
            sentences = tokenize.sent_tokenize(text)
            sentence_embeddings = self.model.encode(sentences)
            return np.mean(sentence_embeddings, axis=0)
        else:
            return np.array(self.model.encode(text))

    def create_corpus_embeddings(self, documents: List[str], sent_tokenize: bool = True) -> np.ndarray:
        if any(doc == '' or doc is None for doc in documents):
            raise ValueError('Corpus contains empty strings.')
        return np.array([self.create_document_embeddings(doc, sent_tokenize) for doc in tqdm(documents)])


class EmbeddingStorage:
    def __init(self):
        pass

    @staticmethod
    def save(embeddings: np.ndarray, file_path: Union[Path, str]):
        file_path = Path(file_path)
        with open(file_path, 'wb') as pkl:
            pickle.dump(embeddings, pkl)

    @staticmethod
    def load(file_path: Union[Path, str]) -> np.ndarray:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'Embeddings file not found at: {file_path}')
        with open(file_path, 'rb') as pkl:
            return pickle.load(pkl)


class EmbeddingPipeline:
    def __init__(
            self,
            model_name: str,
            model_dir: Union[Path, str],
            sent_tokenize: bool = True
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.sent_tokenize = sent_tokenize

        self.model_manager = ModelManager(self.model_name)
        self.model = self.model_manager.load_sentence_transformer(self.model_dir)
        self.embedding_generator = EmbeddingGenerator(self.model)
        self.embedding_storage = EmbeddingStorage()

    def run(
            self,
            documents: List[str],
            embeddings_dir: Union[Path, str],
            file_name: str,
            dim_reduction_params: Optional[dict] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        embeddings = self.embedding_generator.create_corpus_embeddings(documents, self.sent_tokenize)
        embeddings_path = self._build_path(embeddings_dir, file_name, 'embeddings')
        self.embedding_storage.save(embeddings, embeddings_path)

        if dim_reduction_params:
            reduced_embeddings = umap_reduce(embeddings, dim_reduction_params)
            reduced_emebddings_path = self._build_path(embeddings_dir, file_name, 'reduced_embeddings')
            self.embedding_storage.save(reduced_embeddings, reduced_emebddings_path)
        else:
            reduced_embeddings = None

        return embeddings, reduced_embeddings

    def _build_path(self, base_dir: Union[Path, str], file_name: str, suffix: str) -> Path:
        base_dir = Path(base_dir)
        model_suffix = self.model_manager.short_model_name
        return base_dir / f'{file_name}_{suffix}_{model_suffix}.pickle'
