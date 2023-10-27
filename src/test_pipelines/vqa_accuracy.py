from src.test_pipelines import TestPipelineMeta

from src import logger
from src.metrics import print_metrics_results


class VQAAccuracyTestPipeline(TestPipelineMeta):

    __name__ = 'VQA Accuracy Test Pipeline'

    def test(self, *args, **kwargs):

        accuracies_vqa = {}
        accuracies_classic = {}

        for dataset in self.datasets:
            dataset_ = dataset.get_data()
            for model in self.models:
                for metric in self.metrics:
                    answers_ = []
                    for vqa_unit_key in dataset_.keys():
                        answers_.append({
                            'answers': dataset_[vqa_unit_key]['answers'],
                            'model_answers': model.predict(image=dataset_[vqa_unit_key]['image'],
                                                           batch_questions=dataset_[vqa_unit_key]['questions'])})

                    accuracies_vqa[f'm-{model.__name__}_d-{dataset.__name__}'] = \
                        metric(answers_, accuracy_type='vqa')
                    accuracies_classic[f'm-{model.__name__}_d-{dataset.__name__}'] = \
                        metric(answers_, accuracy_type='classic')

        logger.info(print_metrics_results(metric_results=accuracies_vqa,
                                          metric_name='vqa'))
        logger.info(print_metrics_results(metric_results=accuracies_classic,
                                          metric_name='classic'))
