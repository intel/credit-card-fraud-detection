import sys
from abc import ABC

import tritonclient.grpc as triton_grpc

from fraud_detect.serve.config import Config
from fraud_detect.serve.triton.util import is_triton_server_ready


class Pipeline(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.client = triton_grpc.InferenceServerClient(url=f'{config.serve_config.host}:{config.serve_config.port}')
        if not is_triton_server_ready(self.client, config.serve_config.connect_timeout):
            sys.exit()

