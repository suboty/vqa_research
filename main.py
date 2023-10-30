from pathlib import Path

from tqdm import tqdm

from src.metrics.vqa_accuracy import VQAAccuracy
from src.test_pipelines.vqa_accuracy import VQAAccuracyTestPipeline
from src.pipelines.blip import BlipPipeline
from src.dataset_loaders.dummy import DummyDatasetLoader


if __name__ == '__main__':
    # TODO: Add argparse
    dummy_dataset = DummyDatasetLoader(path_to_data=Path('data', 'dummy'))
    dummy_dataset.load_data()

    vqa_accuracy_metric = VQAAccuracy()
    classic_test_pipeline = VQAAccuracyTestPipeline()

    classic_test_pipeline.add_test_dataset(dummy_dataset)
    classic_test_pipeline.add_metric(vqa_accuracy_metric)

    blips = ['Salesforce/blip-image-captioning-large',
             'Salesforce/blip-image-captioning-base',
             'Salesforce/blip-vqa-base',
             'Salesforce/blip-vqa-capfilt-large',
             'Salesforce/blip-itm-base-coco',
             'Salesforce/blip-itm-large-coco',
             'Salesforce/blip-itm-base-flickr',
             'Salesforce/blip-itm-large-flickr']

    for blip_name in tqdm(blips):
        blip_pipeline = BlipPipeline(blip_name)
        classic_test_pipeline.add_model(blip_pipeline)

    # run tests
    classic_test_pipeline.test()
