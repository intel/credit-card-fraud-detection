from typing import Optional, Any

from pydantic import BaseModel


class ServeConfig(BaseModel):
    """serve engine hostname"""
    host: Optional[str] = 'localhost'

    """serve engine port"""
    port: Optional[int] = 8001

    """serve engine connection timeout"""
    connect_timeout: int = 120

    """serve engine name"""
    engine: str = 'triton'


class GradioConfig(BaseModel):
    server_name: str = '0.0.0.0'
    server_port: Optional[int] = None


class AppConfig(BaseModel):
    app_title: str
    # input example dataset file for predict
    input_dataset_example_file: str


class XGBBaseAppConfig(AppConfig):
    xgb_model_name: str
    xgb_model_version: str
    xgb_serve_config_file: str
    gnn_required: bool
    gnn_model_name: Optional[str] = None
    gnn_model_version: Optional[str] = None


class LoytalConfig(BaseModel):
    xgb_model_name: str
    xgb_model_version: str


class Config(BaseModel):
    """serve engine config"""
    serve_config: ServeConfig = ServeConfig()

    """gradio config"""
    gradio_config: Optional[GradioConfig] = GradioConfig()

    """application name used to identity which application should serve"""
    app_name: str = "credit_card"

    """models directory """
    models_path: str = "/workspace/models"

    """application configs"""
    applications: dict[str, dict[str, Any]] = {}

    @staticmethod
    def load(file: str):
        with open(file, "rb") as f:
            import yaml
            return Config(**yaml.safe_load(f))
