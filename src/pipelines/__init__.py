from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Optional, Union

from numpy.typing import ArrayLike

from src import logger

NumpyArrayType = TypeVar("NumpyArrayType", bound=ArrayLike)

PathToModel = TypeVar("PathToModel")


class PipelineMeta(Generic[NumpyArrayType, PathToModel]):

    __name__ = 'Meta Pipeline'

    def __init__(self, path_to_model: Optional[Union[Path, str]] = None):
        try:
            logger.info(f'Downloading of {self.__name__} starts...')
            self.model = self.load_model(path_to_model=path_to_model)
            logger.info(f'Downloading of {self.__name__} is done!')
        except Exception as e:
            logger.error(f'Model {self.__name__} loading is invalid! Error: {e}')
            raise e

    @abstractmethod
    def load_model(self, path_to_model: PathToModel) -> None:
        pass

    @abstractmethod
    def predict(self,
                image: NumpyArrayType,
                batch_questions: NumpyArrayType) -> NumpyArrayType:
        pass

    @abstractmethod
    def __repr__(self):
        return f'Class for {self.__name__} pipeline'
