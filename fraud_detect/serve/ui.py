from typing import Callable

import gradio as gr
import pandas as pd

from fraud_detect.serve.config import Config


def gradio_ui(config: Config, predict: Callable[[pd.DataFrame], pd.DataFrame]) -> gr.Interface:
    app_title = config.applications[config.app_name]['app_title']
    csv_example_file = config.applications[config.app_name]['input_dataset_example_file']
    headers = pd.read_csv(csv_example_file, nrows=0).columns.tolist()
    gr_inputs = [gr.Dataframe(headers=headers,
                              row_count=(1, "dynamic"),
                              col_count=(len(headers), "fixed"),
                              label="Input Dataset",
                              interactive=True)]
    gr_outputs = [
        gr.Dataframe(row_count=(1, "dynamic"),
                     col_count=(1, "fixed"),
                     label="Predictions",
                     headers=["Failures"],
                     datatype=["bool"])]

    gr_examples = pd.read_csv(csv_example_file)

    return gr.Interface(
        predict,
        title=app_title,
        inputs=gr_inputs,
        outputs=gr_outputs,
        examples=[gr_examples.head(3)],
        cache_examples=False
    )
