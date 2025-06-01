from pathlib import Path

import numpy as np

from src.preprocesing.embeddings import EmbeddingStorage
from src.preprocesing.corpus import CorpusBuilder
from src.tuning.grid_search import HDBSCANGridSearch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

DATA_DIR = Path('../data/')
DATA_FILE = 'Projekty energetika zklastrované.xlsx'
PATH_TO_DATA = DATA_DIR / DATA_FILE

MODEL_DIR = Path('../models')
SENTENCE_TRANSFORMER_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SENTENCE_TRANSFORMER_SHORT_MODEL_NAME = SENTENCE_TRANSFORMER_MODEL_NAME.split('/')[-1]

EMBEDDINGS_DIR = Path('../embeddings')
EMBEDDINGS_FILE = f'projects'
REDUCED_EMBEDDINGS_FILE = f'{EMBEDDINGS_DIR}/{EMBEDDINGS_FILE}_reduced_embeddings_{SENTENCE_TRANSFORMER_SHORT_MODEL_NAME}.pickle'

HDBSCAN_PARAMS = {
    'min_cluster_size': range(2, 10),
    'min_samples': range(3, 20),
    'cluster_selection_epsilon': [float(x) for x in np.arange(0.5, 5.0, 0.5)],
    'metric': ['euclidean'],
    'cluster_selection_method': ['eom']
}

LOAD_EMBEDDINGS = True

cb = CorpusBuilder()
cb.load_data_from_file(PATH_TO_DATA)
cb.create_corpus(
    id_col='Kód projektu',
    text_cols=['Název projektu', 'Cíle projektu'],
    join_char='. '
)

embedd_storage = EmbeddingStorage()
reduced_embeddings = embedd_storage.load(REDUCED_EMBEDDINGS_FILE)

gs = HDBSCANGridSearch(HDBSCAN_PARAMS, 'clustering_grid_search_v2')
results = gs.search(reduced_embeddings, ids=cb.corpus['Kód projektu'], documents=cb.corpus['corpus'])
print(results)
