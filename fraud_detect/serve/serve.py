
if __name__ == '__main__':
    import argparse
    from fraud_detect.serve.config import Config
    from fraud_detect.serve.triton.client import run_serve
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, required=True)
    parser.add_argument("-a", "--app", dest="app_name", type=str, required=False, default='credit_card')
    args = parser.parse_args()
    config = Config.load(args.config)
    config.app_name = args.app_name

    run_serve(config)