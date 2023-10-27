import os
from pathlib import Path
from abc import abstractmethod
from typing import List, Generic, TypeVar

from src import logger

DatasetInterface = TypeVar("DatasetInterface")

DatasetParams = TypeVar("DatasetParams")


class DatasetLoaderMeta(Generic[DatasetInterface, DatasetParams]):

    __name__ = 'Meta Dataset Loader'

    def __init__(self,
                 links_for_data: List[str],
                 path_to_data: Path) -> None:

        self.path_to_data = path_to_data

        if os.path.isdir(path_to_data) and os.listdir(path_to_data) != []:
            logger.info(f'Data for {self.__name__} is already download in {path_to_data}')
        else:
            logger.info(f'Downloading of {self.__name__} dataset starts...')
            try:
                self.load_data(links_for_data=links_for_data,
                               path_to_data=path_to_data)
                logger.info(f'Downloading of {self.__name__} dataset is done! Data in {path_to_data}')
            except Exception as e:
                logger.error(f'Dataset {self.__name__} loading is invalid! Error: {e}')
                raise e

    @abstractmethod
    def load_data(self,
                  links_for_data: List[str],
                  path_to_data: Path) -> None:
        pass

    @abstractmethod
    def get_data(self,
                 params: DatasetParams) -> DatasetInterface:
        pass

    @abstractmethod
    def __repr__(self):
        return f'Class for {self.__name__} dataset loading'
