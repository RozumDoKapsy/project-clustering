from pathlib import Path

from src.preprocesing.corpus import CorpusBuilder, CorpusCleaner, CorpusLemmatizerBuilder
from src.preprocesing.embeddings import EmbeddingStorage, EmbeddingPipeline
from src.preprocesing.models import ModelManager
from src.clustering.models import HDBSCANClusteringModel
from src.clustering.pipeline import ClusteringPipeline
from src.clustering.definition import ClusterKeywords, ClusterOAIDefinition

PROGRAMME_LIST = ['TK', 'TS']
LANGUAGE = 'CZ'

DATA_DIR = Path('../data/')
# DATA_FILE = 'raw_projects.csv'
# DATA_FILE = 'TK_TS.xlsx'
DATA_FILE = 'Projekty energetika zklastrované.xlsx'
PATH_TO_DATA = DATA_DIR / DATA_FILE

CORPUS_FILE = 'corpus_projects.csv'
PATH_TO_CORPUS = DATA_DIR / CORPUS_FILE

MODEL_DIR = Path('../models')
SENTENCE_TRANSFORMER_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SENTENCE_TRANSFORMER_SHORT_MODEL_NAME = SENTENCE_TRANSFORMER_MODEL_NAME.split('/')[-1]

UDPIPE_MODEL_NAME = 'czech-pdt-ud-2.5-191206.udpipe'

EMBEDDINGS_DIR = Path('../embeddings')
EMBEDDINGS_FILE = f'projects'
REDUCED_EMBEDDINGS_FILE = f'{EMBEDDINGS_DIR}/{EMBEDDINGS_FILE}_reduced_embeddings_{SENTENCE_TRANSFORMER_SHORT_MODEL_NAME}.pickle'

CLUSTERS_FILE = 'clusters_projects.xlsx'
PATH_TO_CLUSTERS = DATA_DIR / CLUSTERS_FILE

STOPWORDS_FILE = 'stopwords-cs.json'
PATH_TO_STOPWORDS = DATA_DIR / STOPWORDS_FILE

KEYWORDS_FILE = 'clusters_keywords.xlsx'
DEFINITION_FILE = 'clusters_definition.xlsx'

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

    cleaner = CorpusCleaner()
    cleaned_corpus = cleaner.clean(cb.corpus['corpus'])

    lemmatizer = CorpusLemmatizerBuilder(
        model_dir=MODEL_DIR,
        model_name=UDPIPE_MODEL_NAME,
        stopwords_path=PATH_TO_STOPWORDS
    ).build()

    lemmatized_corpus = lemmatizer.process(cleaned_corpus)

    if LOAD_EMBEDDINGS:
        embedd_storage = EmbeddingStorage()
        reduced_embeddings = embedd_storage.load(REDUCED_EMBEDDINGS_FILE)
    else:
        embeddings_pipeline = EmbeddingPipeline(
            model_name=SENTENCE_TRANSFORMER_MODEL_NAME,
            model_dir=MODEL_DIR
        )
        embeddings, reduced_embeddings = embeddings_pipeline.run(
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

    definition = ClusterKeywords()
    definition.fit(
        documents=lemmatized_corpus,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm='l1'
    )
    definition.extract_cluster_keywords(
        labels=pipeline.model.labels,
        topn=10
    )
    clusters_keywords_df = definition.get_clusters_keywords_df()
    clusters_keywords_df.to_excel(DATA_DIR / KEYWORDS_FILE, index=False)

    oai_definition = ClusterOAIDefinition()
    oai_definition.get_clusters_definition(
        clusters_keywords=definition.cluster_kw,
        labels=pipeline.model.labels,
        documents=cb.corpus['corpus']
    )

    clusters_definition_df = oai_definition.get_clusters_definition_df()
    clusters_definition_df.to_excel(DATA_DIR / DEFINITION_FILE, index=False)

    # TODO: result = Clustering Metadata (metrics), Cluster information (silhouette score, definition), ClusteredDocuments)

    print(pipeline.model.evaluation_results)


if __name__ == '__main__':
    main()
