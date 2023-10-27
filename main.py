from pathlib import Path


from src import logger

from src.metrics.vqa_accuracy import VQAAccuracy
from src.test_pipelines.vqa_accuracy import VQAAccuracyTestPipeline
from src.pipelines.dummy import DummyPipeline
from src.dataset_loaders.dummy import DummyDatasetLoader


if __name__ == '__main__':
    dummy_dataset = DummyDatasetLoader(path_to_data=Path('data', 'dummy'))
    dummy_dataset.load_data()

    dummy_metric = VQAAccuracy()
    dummy_pipeline = DummyPipeline()

    dummy_test_pipeline = VQAAccuracyTestPipeline()
    dummy_test_pipeline.add_test_dataset(dummy_dataset)
    dummy_test_pipeline.add_metric(dummy_metric)
    dummy_test_pipeline.add_model(dummy_pipeline)

    dummy_test_pipeline.test()
