from fraud_detect.serve.config import Config


from fraud_detect.serve.ui import gradio_ui


def run_serve(config: Config):
    if config.app_name in ['credit_card', 'airbnb']:
        from fraud_detect.serve.triton.pipeline.xgb_pipeline import XGBBasedPipeline
        pipeline = XGBBasedPipeline(config)
    elif config.app_name == 'loytal':
        from fraud_detect.serve.triton.pipeline.loytal_pipeline import LoytalPipeline
        pipeline = LoytalPipeline(config)
    else:
        raise ValueError(f"app name {config.app_name} is invalid")

    interface = gradio_ui(config, lambda data: pipeline(data))
    interface.launch(
        server_name=config.gradio_config.server_name,
        server_port=config.gradio_config.server_port,
        share=True
    )
