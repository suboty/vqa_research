from pathlib import Path


from src.metrics.vqa_accuracy import VQAAccuracy
from src.test_pipelines.vqa_accuracy import VQAAccuracyTestPipeline
from src.pipelines.dummy import DummyPipeline
from src.pipelines.blip import BlipPipeline
from src.dataset_loaders.dummy import DummyDatasetLoader


if __name__ == '__main__':
    # TODO: Add argparse
    dummy_dataset = DummyDatasetLoader(path_to_data=Path('data', 'dummy'))
    dummy_dataset.load_data()

    dummy_metric = VQAAccuracy()

    dummy_pipeline = DummyPipeline()
    blip_pipeline = BlipPipeline('Salesforce/blip-vqa-capfilt-large')

    dummy_test_pipeline = VQAAccuracyTestPipeline()

    # create test pipeline
    dummy_test_pipeline.add_test_dataset(dummy_dataset)
    dummy_test_pipeline.add_metric(dummy_metric)
    dummy_test_pipeline.add_model(blip_pipeline)

    # run tests
    dummy_test_pipeline.test()
