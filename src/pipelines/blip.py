from abc import ABC
from typing import List

import numpy as np
from transformers import AutoProcessor, BlipForQuestionAnswering

from src import logger
from src.pipelines import PipelineMeta, ModelError


class BlipPipeline(PipelineMeta, ABC):

    __name__ = '__TYPE__'

    def __init__(self, path_to_model):
        self.processor = None
        self.__name__ = self.__name__.replace('__TYPE__', path_to_model.split('/')[1])
        super().__init__(path_to_model)

    def load_model(self, path_to_model: str) -> None:
        model = BlipForQuestionAnswering.from_pretrained(path_to_model)
        self.processor = AutoProcessor.from_pretrained(path_to_model)
        return model

    def predict(self, image: np.array, batch_questions: np.array) -> List[np.array]:
        batch_answers = []
        for question in batch_questions:
            try:
                inputs = self.processor(images=image, text=question, return_tensors="pt")
                batch_answers.append(self.processor.decode(self.model.generate(**inputs)[0],
                                                           skip_special_tokens=True))
            except Exception as e:
                logger.error(f'BLIP model {self.__name__} is aborted! Error: {e}')
                raise ModelError(f'BLIP model {self.__name__} is aborted! Error: {e}')

        return batch_answers
