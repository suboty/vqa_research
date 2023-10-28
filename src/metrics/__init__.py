import re
from abc import abstractmethod
from typing import TypeVar, Dict

MetricType = TypeVar("MetricType")


class MetricMeta:

    __name__ = 'Meta Metric'

    @abstractmethod
    def __call__(self, *args, **kwargs) -> MetricType:
        pass

    @abstractmethod
    def __repr__(self):
        return f'Class for {self.__name__} metric'


regexp_dataset_name = re.compile(r'(?=d-([A-Za-z\s-]*))')
regexp_model_name = re.compile(r'(?=m-([A-Za-z\s-]*))')


def print_metrics_results(metric_results: Dict,
                          metric_name: str):
    answer_repr = ''
    answer_repr_ = {}
    for i, metric_result_key in enumerate(metric_results.keys()):
        from src import logger
        logger.info(metric_result_key)
        answer_repr_[i] = f'{metric_name.upper()}: ' \
                          f'{regexp_model_name.findall(metric_result_key)[0]} | ' \
                          f'{regexp_dataset_name.findall(metric_result_key)[0]} | ' \
                          f'{metric_results[metric_result_key]:.2f}'

    for key in list(answer_repr_.keys()):
        answer_repr += answer_repr_[key]

    return answer_repr
