import time

from loguru import logger
from tritonclient import utils as triton_utils


# Check to see if server is live
def is_triton_server_ready(client, connection_timeout=60):
    server_start = time.time()
    while True:
        try:
            if client.is_server_ready():
                return True
        except triton_utils.InferenceServerException:
            pass
        if time.time() - server_start > connection_timeout:
            logger.error('triton server not ready, please check logs for more information.')
            return False
        time.sleep(1)
