from pathlib import Path
import pandas as pd

from typing import List, Union, Optional


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
