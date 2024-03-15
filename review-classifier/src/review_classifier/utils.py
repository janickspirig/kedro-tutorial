from typing import Any, Dict, Optional, Union

import pandas as pd
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from review_classifier import settings

configure_project(settings.BASE_PATH)


class ConfigInterface:
    """Exposes functions to interact with Kedro catalog and parameters."""
    def __init__(self, env: Optional[str] = "base"):
        self._config_loader = settings.CONFIG_LOADER_CLASS(
            conf_source=settings.CONF_PATH
        )

        with KedroSession.create(
            project_path=settings.PROJECT_PATH, env=env
        ) as session:
            self._kedro_context = session.load_context()

    def load_data_from_catalog(
        self,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Loads dataset(s) from the catalog.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset to be loaded. Defaults to None, i.e., meaning all datasets will be loaded.

        Returns:
            Dictionary containing loaded datasets or single dataset.
        """
        if dataset_name:
            return self._kedro_context.catalog.load(dataset_name)
        
        return {
            name: dataset.load()
            for name, dataset in self._kedro_context.catalog.datasets.__dict__.items()
        }

    def load_params(
        self, key: Optional[str] = None
    ) -> Union[int, str, float, Dict[str, Union[int, str, float]]]:
        """Loads and returns parameters from the parameters file.

        Args:
            key (Optional[str], optional): Value of key under which parameter should be returned. Defaults to None, meaning all parameters will be loaded.

        Returns:
            Collections of parameters stored in dictionary or single parameter.
        """
        params = self._kedro_context.catalog.load("parameters")
        return params.get(key, params)

    def save_single_dataset(self, dataset_name: str, data: Any):
        """Saves a single dataset to the catalog.

        Args:
            dataset_name (str): Name of the dataset, must match a key in the catalog.
            data (Any): Data to be stored
        """
        self._kedro_context.catalog.save(name=dataset_name, data=data)
