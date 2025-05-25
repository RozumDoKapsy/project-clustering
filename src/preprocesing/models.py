from pathlib import Path
from sentence_transformers import SentenceTransformer
from corpy.udpipe import Model as UDPipeModel

import pickle
from typing import Union


class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.short_model_name = model_name.split('/')[-1]
        self.model = None

    def _get_model_path(self, model_dir: Union[Path, str]) -> Path:
        return Path(model_dir) / f'{self.short_model_name}.pickle'

    @staticmethod
    def _load_from_file(file_path: Path) -> SentenceTransformer:
        with open(file_path, 'rb') as pkl:
            return pickle.load(pkl)

    def _load_from_huggingface(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    def load_sentence_transformer(self, model_dir: Union[Path, str]) -> SentenceTransformer:
        file_path = self._get_model_path(model_dir)
        if file_path.exists():
            self.model = self._load_from_file(file_path)
            return self.model
        else:
            try:
                self.model = self._load_from_huggingface()
                return self.model
            except Exception as e:
                raise ValueError(f'SentenceTransformer model {self.model_name} could not be loaded from HuggingFace.') from e

    def load_udpipe_model(self, model_dir: Union[Path, str]) -> UDPipeModel:
        file_path = Path(model_dir) / f'{self.short_model_name}'
        if file_path.exists():
            self.model = UDPipeModel(file_path)
            return self.model
        else:
            raise FileNotFoundError(f'UDPipe model was not found at location: {file_path}')

    def save(self, model_dir: Union[Path, str]):
        if self.model is None:
            raise ValueError(f'No model loaded.')
        file_path = self._get_model_path(model_dir)
        with open(file_path, 'wb') as pkl:
            pickle.dump(self.model, pkl)


