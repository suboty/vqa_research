import os
import json
from abc import ABC
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from src.dataset_loaders import DatasetLoaderMeta


class DummyDatasetLoader(DatasetLoaderMeta, ABC):
    __name__ = 'Dummy Dataset'

    def __init__(self,
                 links_for_data: Optional[List[str]] = None,
                 path_to_data: Optional[Path] = None
                 ) -> None:
        super().__init__(links_for_data=links_for_data,
                         path_to_data=path_to_data)

        self.qa = {}

    def load_data(self) -> None:

        with open(Path(self.path_to_data, 'qa', 'qa.json'), 'r') as qa_file:
            self.qa = json.load(qa_file)

    def get_data(self,
                 params: Optional[Dict] = None) -> Dict:
        dataset = {}

        for i, image_path in enumerate(os.listdir(Path(self.path_to_data, 'images'))):
            dataset[i] = {'image_path': image_path,
                          'image': np.asarray(Image.open(Path(self.path_to_data, 'images', image_path))),
                          'questions': self.qa[image_path]['questions'],
                          'answers': self.qa[image_path]['answers']}

        return dataset
