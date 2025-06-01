import re

import mlflow
from mlflow.tracking import MlflowClient

import hashlib

from typing import Dict, Any


def create_run_name(prefix: str, params: Dict[str, Any]) -> str:
    params_text = '_'.join(f'{k}={v}' for k, v in params.items())
    hash_str = hashlib.md5('_'.join(params_text).encode()).hexdigest()[:6]
    return f'{prefix}_{hash_str}'


class MlFlowManager:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri('http://localhost:5000')  # TODO: refactor
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

        self.experiment = self.client.get_experiment_by_name(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        self.next_version_tag = self._get_next_version_tag()

    def _get_next_version_tag(self):
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="tags.version LIKE 'v%'",
            order_by=["attributes.start_time DESC"]
        )

        max_version = 0
        for r in runs:
            version_tag = r.data.tags.get('version')
            if version_tag:
                match = re.match(r'v(\d)', version_tag)
                if match:
                    version_num = int(match.group(1))
                    max_version = max(version_num, max_version)

        return f'v{max_version + 1}'
