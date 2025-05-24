from pathlib import Path

from src.preprocesing.corpus import CorpusBuilder
from src.preprocesing.embeddings import EmbeddingStorage, EmbeddingPipeline
from src.clustering.models import HDBSCANClusteringModel
from src.clustering.pipeline import ClusteringPipeline

PROGRAMME_LIST = ['TK', 'TS']
LANGUAGE = 'CZ'

DATA_DIR = Path('../data/')
# DATA_FILE = 'raw_projects.csv'
DATA_FILE = 'TK_TS.xlsx'
PATH_TO_DATA = DATA_DIR / DATA_FILE

CORPUS_FILE = 'corpus_projects.csv'
PATH_TO_CORPUS = DATA_DIR / CORPUS_FILE

MODEL_DIR = Path('../models')
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SHORT_MODEL_NAME = MODEL_NAME.split('/')[-1]

EMBEDDINGS_DIR = Path('../embeddings')
EMBEDDINGS_FILE = f'projects'
REDUCED_EMBEDDINGS_FILE = f'{EMBEDDINGS_DIR}/{EMBEDDINGS_FILE}_reduced_embeddings_{SHORT_MODEL_NAME}.pickle'

CLUSTERS_FILE = 'clusters_projects.xlsx'
PATH_TO_CLUSTERS = DATA_DIR / CLUSTERS_FILE

UMAP_PARAMS = {
    'n_neighbors': 5,
    'min_dist': 0.001,
    'n_components': 50,
    'metric': 'cosine'
}

HDBSCAN_PARAMS = {
    'min_cluster_size': 4,
    'min_samples': 5,
    'cluster_selection_epsilon': 0.05,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom'
}

LOAD_EMBEDDINGS = True


def main():
    cb = CorpusBuilder()
    cb.load_data_from_file(PATH_TO_DATA)
    cb.create_corpus(
        id_col='Kód projektu',
        text_cols=['Název projektu', 'Cíle projektu'],
        join_char='. '
    )
    cb.save(PATH_TO_CORPUS)

    if LOAD_EMBEDDINGS:
        embedd_storage = EmbeddingStorage()
        reduced_embeddings = embedd_storage.load(REDUCED_EMBEDDINGS_FILE)
    else:
        embeddings_pipeline = EmbeddingPipeline(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR
        )
        reduced_embeddings = embeddings_pipeline.run(
            documents=cb.corpus['corpus'],
            embeddings_dir=EMBEDDINGS_DIR,
            file_name=EMBEDDINGS_FILE,
            dim_reduction_params=UMAP_PARAMS
        )

    pipeline = ClusteringPipeline(HDBSCANClusteringModel(HDBSCAN_PARAMS))
    pipeline.run(
        embeddings=reduced_embeddings,
        ids=cb.corpus['Kód projektu'],
        documents=cb.corpus['corpus'],
        save=True,
        file_path=PATH_TO_CLUSTERS
    )
    print(pipeline.model.evaluation_results)

    # TODO: result = Clustering Metadata (metrics), Cluster information (silhouette score, definition), ClusteredDocuments)


if __name__ == '__main__':
    main()
