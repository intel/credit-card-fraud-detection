
if __name__ == '__main__':
    import argparse
    from fraud_detect.serve.config import Config
    from fraud_detect.serve.triton.client import run_serve
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    run_serve(config)