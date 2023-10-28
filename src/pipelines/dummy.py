import random
from abc import ABC
from typing import List

import numpy as np

from src.pipelines import PipelineMeta


class DummyPipeline(PipelineMeta, ABC):

    __name__ = 'Dummy Pipeline'

    __answers = ["Yes", "No", "A dog"]

    def load_model(self, path_to_model: str) -> None:
        def __dummy_model(image: np.array, batch_questions: np.array):
            return [random.choice(self.__answers) for _ in batch_questions]
        return __dummy_model

    def predict(self, image: np.array, batch_questions: np.array) -> List[np.array]:
        return self.model(image=image,
                          batch_questions=batch_questions)
