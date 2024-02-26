import os

import pandas as pd
import xgboost as xgb
import yaml

from fraud_detect.preprocessing.data_preprocess import XGBoostPreprocessor
from fraud_detect.serve.config import Config, XGBBaseAppConfig
from fraud_detect.serve.triton.pipeline.base import Pipeline


class XGBBasedPipeline(Pipeline):
    def __init__(self, config: Config):
        super().__init__(config)
        self.pipeline_config = XGBBaseAppConfig(**config.applications[config.app_name])
        self.model = xgb.Booster()
        xgb_model_uri = os.path.join(self.config.models_path,
                                     f"{self.pipeline_config.xgb_model_name}/{self.pipeline_config.xgb_model_version}/xgboost.json")
        self.model.load_model(xgb_model_uri)

        with open(self.pipeline_config.xgb_serve_config_file, 'r') as file:
            self.serve_config = yaml.safe_load(file)

        self.preprocessor = XGBoostPreprocessor(self.serve_config['data_processing'], self.model.feature_names)

        try:
            self.threshold = self.serve_config['classification_threshold']
        except:
            self.threshold = 0.5

    def __call__(self, input: pd.DataFrame) -> pd.DataFrame:
        # step 1 data preprocess
        import tritonclient.grpc as triton_grpc

        if self.pipeline_config.gnn_required:
            # step2 gnn process
            triton_input_gnn = triton_grpc.InferInput('input__0', input.to_numpy().shape, 'BYTES')
            triton_input_gnn.set_data_from_numpy(input.to_numpy())
            triton_output_gnn = triton_grpc.InferRequestedOutput('output__0')
            gnn_response = self.client.infer(
                self.pipeline_config.gnn_model_name,
                model_version=self.pipeline_config.gnn_model_version,
                inputs=[triton_input_gnn],
                outputs=[triton_output_gnn]
            )
            gnn_response = gnn_response.as_numpy("output__0")
            # step 3 final xgb
            # TODO Dataprocess for the GNN output
            #gnn_data = self.processor.preprocess_gnn_data(gnn_response)
            gnn_arr = gnn_response.astype('float32')
            triton_input = triton_grpc.InferInput('input__0', gnn_arr.shape, 'FP32')
            triton_input.set_data_from_numpy(gnn_arr)
            triton_output = triton_grpc.InferRequestedOutput('output__0')
            response = self.client.infer(
                self.pipeline_config.xgb_model_name,
                model_version=self.pipeline_config.xgb_model_version,
                inputs=[triton_input],
                outputs=[triton_output]
            )
        else:
            data = self.preprocessor.preprocess_data(input=input)
            arr = data.to_numpy().astype('float32')
            triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
            triton_input.set_data_from_numpy(arr)
            triton_output = triton_grpc.InferRequestedOutput('output__0')
            # step2 baseline xgb
            response = self.client.infer(
                self.pipeline_config.xgb_model_name,
                model_version=self.pipeline_config.xgb_model_version,
                inputs=[triton_input],
                outputs=[triton_output]
            )

        np_data = pd.DataFrame(response.as_numpy('output__0'))
        result_df = np_data[0] >= self.threshold
        result_df = result_df.to_frame(name='predicted_class')
        return result_df
