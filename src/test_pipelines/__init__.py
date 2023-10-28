from abc import abstractmethod
from typing import Generic, TypeVar, Callable

from src import logger

DatasetType = TypeVar("DatasetType")

ModelType = TypeVar("ModelType")


class TestPipelineMeta(Generic[DatasetType, ModelType]):

    __name__ = 'Meta Test Pipeline'

    def __init__(self):
        self.datasets = []
        self.models = []
        self.metrics = []

    @abstractmethod
    def add_test_dataset(self, dataset: DatasetType):
        self.datasets.append(dataset)
        logger.debug(f'Dataset {dataset.__name__ if dataset.__name__ else ""} is appended to test datasets')

    @abstractmethod
    def add_model(self, model: ModelType):
        self.models.append(model)
        logger.debug(f'Model {model.__name__ if model.__name__ else ""} is appended to models for testing')

    @abstractmethod
    def add_metric(self, metric: Callable):
        self.metrics.append(metric)
        logger.debug(f'Metric {metric.__name__ if metric.__name__ else ""} is appended to test metrics')

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        return f'Class for {self.__name__} test pipeline'
