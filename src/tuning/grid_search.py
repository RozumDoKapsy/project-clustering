from abc import ABC
from itertools import product
import os
from pathlib import Path
import shutil

import json

import numpy as np

from tqdm import tqdm

import mlflow

from src.clustering.models import BaseClusteringModel, HDBSCANClusteringModel, AgglomerativeClusteringModel
from src.clustering.pipeline import ClusteringPipeline
from src.tuning.mlflow_utils import MlFlowManager, create_run_name
from src.utils.processing_utils import umap_reduce

from typing import Dict, List, Any, Type, Union, Optional


class GridSearch(ABC):
    def __init__(
            self,
            params: Dict[str, Any],
            experiment_name: Optional[str] = None,
            model_class: Optional[Type[BaseClusteringModel]] = None,
            run_name_prefix: Optional[str] = None
    ):
        self.params = params
        self.path_to_tmp_artifacts = self._create_tmp_dir()

        self.model_class = model_class
        self.run_name_prefix = run_name_prefix
        self.mlflow_manager = MlFlowManager(experiment_name) if experiment_name else None

    @staticmethod
    def _create_tmp_dir() -> Path:
        path_to_tmp = Path('tmp_artifacts')
        if not path_to_tmp.exists():
            os.mkdir('tmp_artifacts')
        return path_to_tmp

    def _get_params_grid(self) -> List[Dict[str, Any]]:
        param_names = list(self.params.keys())
        param_values = list(self.params.values())
        return [dict(zip(param_names, values)) for values in product(*param_values)]

    @staticmethod
    def _log_params(params: Dict[str, Any], additional_params: Optional[Dict[str, Any]] = None):
        mlflow.log_params(params)
        if additional_params:
            mlflow.log_params({f'UMAP__{k}': v for k, v in additional_params.items()})  # TODO: variable prefix

    @staticmethod
    def _log_evaluation_metrics(evaluation_results: Dict[str, Any], skip_keys: Optional[List[str]] = None):
        clustering_metrics = {k: v for k, v in evaluation_results.items()
                              if k not in skip_keys}
        mlflow.log_metrics(clustering_metrics)

    def _log_artifacts(self):
        mlflow.log_artifacts(self.path_to_tmp_artifacts)

    def search(
            self,
            embeddings: np.ndarray,
            ids: List[Union[str, int]],
            documents: List[str],
            additional_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        grid = self._get_params_grid()
        evaluation_results = []

        for idx, params in tqdm(enumerate(grid)):
            path_to_results = self.path_to_tmp_artifacts / 'clustering_results.csv'
            clustering_pipeline = ClusteringPipeline(self.model_class(params))
            clustering_pipeline.run(
                embeddings=embeddings,
                ids=ids,
                documents=documents,
                save=True,
                file_path=path_to_results
            )

            clusters_sil_score = 'clusters_silhouette_scores'
            clusters_metrics_file = f'{clusters_sil_score}.json'

            clusters_metrics = clustering_pipeline.model.evaluation_results.get(clusters_sil_score)
            with open(self.path_to_tmp_artifacts / clusters_metrics_file, 'w') as f:
                json.dump(clusters_metrics, f)

            if self.mlflow_manager:
                with mlflow.start_run(
                        run_name=create_run_name(self.run_name_prefix, params),
                        tags={'version': self.mlflow_manager.next_version_tag}
                ):
                    self._log_params(params, additional_params)
                    self._log_evaluation_metrics(
                        evaluation_results=clustering_pipeline.model.evaluation_results,
                        skip_keys=[clusters_sil_score]
                    )
                    self._log_artifacts()

            evaluation_results.append({
                'grid_index': idx,
                'parameters': params,
                'evaluation': clustering_pipeline.model.evaluation_results
            })

        if self.path_to_tmp_artifacts.exists():
            shutil.rmtree(self.path_to_tmp_artifacts)
        return evaluation_results


class HDBSCANGridSearch(GridSearch):
    def __init__(self, params: Dict[str, Any], experiment_name: Optional[str] = None):
        super().__init__(params=params, experiment_name=experiment_name, model_class=HDBSCANClusteringModel,
                         run_name_prefix='HDBSCAN')


class AgglomerativeClusteringGridSearch(GridSearch):
    def __init__(self, params: Dict[str, Any], experiment_name: Optional[str] = None):
        super().__init__(params=params, experiment_name=experiment_name, model_class=AgglomerativeClusteringModel,
                         run_name_prefix='Agglomerative')


class UMAPGridSearch(GridSearch):
    def __init__(self, params: Dict[str, Any], clustering_search: Type[GridSearch], clustering_params: Dict[str, Any]):
        super().__init__(params)
        self.clustering_search = clustering_search
        self.clustering_params = clustering_params

    def search(
            self,
            embeddings: np.ndarray,
            ids: List[Union[str, int]],
            documents: List[str],
            additional_params: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        grid = self._get_params_grid()
        evaluation_results = []

        for idx, umap_params in tqdm(enumerate(grid)):
            reduced_embeddings = umap_reduce(embeddings, umap_params)

            clustering_searcher = self.clustering_search(self.clustering_params)
            results = clustering_searcher.search(reduced_embeddings, ids, documents, umap_params)

            evaluation_results.append({
                'grid_index': idx,
                'parameters': umap_params,
                'evaluation': results
            })
        return evaluation_results
