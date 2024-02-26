import pandas as pd

from fraud_detect.serve.config import Config, LoytalConfig
from fraud_detect.serve.triton.pipeline.pipeline import Pipeline


class LoytalPipeline(Pipeline):

    def __init__(self, config: Config):
        super().__init__(config)
        self.pipeline_config = LoytalConfig(**config.applications[config.app_name])

    def __call__(self, input: pd.DataFrame) -> pd.DataFrame:

        return input
