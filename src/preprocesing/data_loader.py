from pathlib import Path
import pandas as pd

from typing import List, Union


class DataLoader:
    def __init__(self, programme_list: List[str], language: str):
        self.programme_list = programme_list
        self.language = language
        self.data = pd.DataFrame()

    def get_isvav_projects(self) -> pd.DataFrame:
        """  """
        df = pd.read_csv("https://www.isvavai.cz/dokumenty/opendata/CEP-projekty.csv")
        filtered_df = self._filter_programme(df)
        filtered_df = self._extract_columns(filtered_df)
        self.data = filtered_df
        return self.data

    def _filter_programme(self, df: pd.DataFrame) -> pd.DataFrame:
        df_programme_list = df['kod_programu'].unique()
        missing_programmes = [i for i in self.programme_list if i not in df_programme_list]
        if missing_programmes:
            raise ValueError(f'Programmes {missing_programmes} not in extracted data.')
        return df[df['kod_programu'].isin(self.programme_list)]

    def _extract_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.language.lower() == 'cz':
            cols = ['kod_projektu', 'kod_programu', 'nazev_projektu_originalni', 'cile_reseni_originalni']
        elif self.language.lower() == 'en':
            cols = ['kod_projektu', 'kod_programu', 'nazev_projektu_anglicky', 'cile_reseni_anglicky',
                    'klicova_slova_anglicky']
        else:
            raise ValueError(f'Language {self.language} not supported.')
        return df[cols]

    def save_data(self, file_path: Union[Path, str]):
        file_path = Path(file_path)
        if self.data is None or self.data.empty:
            raise ValueError('Data is empty.')

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix == '.csv':
            self.data.to_csv(file_path, index=False)
        elif file_path.suffix == '.xlsx':
            self.data.to_excel(file_path, index=False)
        else:
            raise ValueError(f'Unsupported file type: {file_path.suffix}.')



