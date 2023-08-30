"""
Copyright [2022-23] [Intel Corporation]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python
# coding: utf-8
import argparse
import os, time, gc, sys, glob
import pandas as pd 
import numpy as np
import yaml 
import time 
from src.utils.data_utils import *
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import shutil

very_start = time.time()

PATH_HOME = os.path.dirname(os.path.realpath(__file__))

class WFProcessor:

    def __init__(self, cfg):
        try:
            self.tmp_path = "/cnvrg/tmp"
            self.log_path = os.path.join(self.tmp_path, 'logs')
            os.makedirs(self.log_path, exist_ok=True)
            self.model_save_path = os.path.join(self.tmp_path, 'models')
            os.makedirs(self.model_save_path, exist_ok=True)
            self.ray_params = None
            self.worker_ips = None
            self.in_memory = False
            self.train_framework = "pandas"
            self.test_backend = "xgboost-native"
            self.has_training = False
            if cfg.input_path is None or cfg.input_path == "":
                self.raw_data_path = self.fetch_data("/input/data_connector/fraud_detection/dataset")
            else:
                self.raw_data_path = self.fetch_data(cfg.input_path)
            self.raw_data_format = "csv"
            self.dp_framework = "pandas"
            if cfg.output_path is None or cfg.output_path == "":
                scripts_dir = Path(__file__).parent.resolve()
                output_path = os.path.join(scripts_dir, "/output")
                self.processed_data_path = output_path
            else:
                self.processed_data_path = cfg.output_path
            os.makedirs(self.processed_data_path, exist_ok=True)
            self.processed_data_format = "csv"
            if cfg.config_file is None or cfg.config_file == "":
                scripts_dir = Path(__file__).parent.resolve()
                config_file = os.path.join(scripts_dir, "config.yaml")
                self.read_data_processing_steps(config_file)
            else:
                self.read_data_processing_steps(cfg.config_file)
            self.dp_engine = "pandas"
            self.has_dp = True
        except Exception as e:
            print('Failed to read data preprocessing steps. This is either due to wrong parameters defined in the config file as shown: '+ str(e)
                    + ' or there is no need for data preprocessing.')
            print("Program End.")
            sys.exit()

    def fetch_data(self,dataset_path):

        if dataset_path.startswith("https://"):
            a = urlparse(dataset_path)
            output_dir = "/tmp/"
            to_save = os.path.join(output_dir, os.path.basename(a.path))
            if not os.path.exists(to_save):
                with requests.get(dataset_path, stream=True) as r:
                    # check header to get content length, in bytes
                    total_length = int(r.headers.get("Content-Length"))
                    # implement progress bar via tqdm
                    with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                        # save the output to a file
                        with open(f"{to_save}", 'wb') as output:
                            shutil.copyfileobj(raw, output)
        else:
            to_save = dataset_path

        # *** Read Data ***
        if not os.path.exists(to_save):
            raise FileNotFoundError(f"{to_save} is not exists.")

        print(f"Data is fetched to {to_save}")
        return to_save

    def read_data_processing_steps(self, dp_config_file):
        with open(dp_config_file, 'r') as file:
            dp_steps = yaml.safe_load(file)

        self.pre_splitting_steps = dp_steps['pre_splitting_transformation']
        self.data_splitting_rule = dp_steps['data_splitting']
        self.post_splitting_steps = dp_steps['post_splitting_transformation']

    def read_raw_data(self):
        print('reading raw data...')
        if self.raw_data_format == 'csv':
            self.data = read_csv_files(self.raw_data_path, engine=self.dp_engine)

    def pre_splitting_transform(self):
        print("transform pre-splitting data...")
        if self.dp_engine != 'spark':
            from src.preprocessing.pandas.pre_splitting_transformation import PreSplittingTransformer
            pre_splitting_transformer = PreSplittingTransformer(self.data, self.pre_splitting_steps, self.dp_engine)
            self.data = pre_splitting_transformer.process()
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")

    def split_data(self):
        print('splitting data...')
        if self.dp_engine != 'spark':
            import pandas as pd
            from src.preprocessing.pandas.data_splitting import DataSplitter
            data_splitter = DataSplitter(self.data, self.data_splitting_rule)
            self.train_data, self.test_data = data_splitter.process()
            self.data = None
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")

    def post_splitting_transform(self):
        print("transform pre-splitting data...")
        if self.dp_engine != 'spark':
            from src.preprocessing.pandas.post_splitting_transformation import PostSplittingTransformer
            pre_splitting_transformer = PostSplittingTransformer(self.train_data, self.test_data, self.post_splitting_steps, self.dp_engine)
            self.train_data, self.test_data = pre_splitting_transformer.process()
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")

    def save_processed_data(self):
        print('saving data...')
        if self.dp_engine != 'spark':
            if self.dp_engine == 'modin':
                import modin.pandas as pd
            else:
                import pandas as pd
            if self.processed_data_format == 'csv':
                data = pd.concat([self.train_data, self.test_data])
                data.to_csv(self.processed_data_path+'/processed_data.csv', index=False)
                print(f'data saved under the path {self.processed_data_path}/processed_data.csv')
        else:
            raise NotImplementedError("currently only pandas-based data preprocessing is supported")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input-path",
            required=False,
            type=str,
            help="specify the input dataset file path")
    parser.add_argument(
            "--output-path",
            required=False,
            type=str,
            help="specify the output file path")
    parser.add_argument(
            "--config-file",
            required=False,
            type=str,
            help="specify the config file name")

    args, _ = parser.parse_known_args()
    wf_processor = WFProcessor(args)
    
    start = time.time()
    print("prepare env took %.1f seconds" % ((time.time()-start)))

    dp_start = time.time()
    wf_processor.read_raw_data()
    print("dp read data took %.1f seconds" % ((time.time()-start)))
    start = time.time()
    wf_processor.pre_splitting_transform()
    print("dp transform pre-splitting data took %.1f seconds" % ((time.time()-start)))
    start = time.time()
    wf_processor.split_data()
    print("dp split data took %.1f seconds" % ((time.time()-start)))
    start = time.time()
    wf_processor.post_splitting_transform()
    print("dp transform post-splitting data took %.1f seconds" % ((time.time()-start)))
    start = time.time()
    wf_processor.save_processed_data()
    print("dp save data took %.1f seconds" % ((time.time()-start)))
    print("data preprocessing took %.1f seconds" % ((time.time()-dp_start)))
    print('The whole workflow processing took %.1f seconds'%(time.time()-very_start))
