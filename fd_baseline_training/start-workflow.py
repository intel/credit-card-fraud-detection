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
            self.worker_ips = None
            self.dp_engine = 'pandas'
            self.train_data_path = cfg.input_path
            self.train_data_format = "csv"
            self.train_framework = "pandas"
            self.test_backend = "xgboost-native"
            if cfg.config_file == "":
                scripts_dir = Path(__file__).parent.resolve()
                config_file = os.path.join(scripts_dir, "config.yaml")
                self.read_training_configs(config_file)
            else:
                self.read_training_configs(cfg.config_file)
            self.ray_params = None
            self.in_memory = False
        except Exception as e: 
            print('Failed to read model training configurations. This is either due to wrong parameters defined in the config file as shown: '+ str(e) 
                    + ' or there is no need for model training.')
            print("Program End.")
            sys.exit()
    
    
    def read_training_configs(self, train_config_file):
        with open(train_config_file, 'r') as file:
            train_configs = yaml.safe_load(file)
        
        self.train_data_spec = train_configs['data_spec']
        try:
            self.hpo_spec = train_configs['hpo_spec']
            self.hpo_needed = True
        except:
            self.hpo_spec = None 
            self.hpo_needed = False
            print("no need for HPO")
        try:
            self.train_model_spec = train_configs['model_spec']
            self.hpo_needed = False
        except:
            self.train_model_spec = None 
            self.hpo_needed = True
            print("no need for training")
        
        if self.hpo_spec is None and self.train_model_spec is None:
            print("none of the hpo_spec and model_spec is specified. Program End.")
            sys.exit()
        elif self.hpo_spec is not None and self.train_model_spec is not None:
            print("Pls specify either hpo_spec or model_spec. Both are not accepted. Program End.")

    def read_train_data(self):
        print('reading training data...')
        if self.train_data_format == 'csv':
            self.data = read_csv_files(self.train_data_path, engine='pandas')

                    
    def train_model(self, df):
        print('start training models soon...')
        from src.training.pandas.model_training import Trainer
        trainer = Trainer(self.train_data_spec, df, self.train_model_spec, self.test_backend, self.in_memory, self.tmp_path, self.worker_ips, self.ray_params, self.hpo_spec)

        if self.hpo_needed:
            trainer.run_hpo()
        else:
            trainer.process()
            trainer.save_model(self.model_save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        required=True,
        default="/input/preprocess/output",
        type=str,
        help="speficy the preprocess file name")
    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="speficy the config file name")
    
    args, _ = parser.parse_known_args()
    wf_processor = WFProcessor(args)
    
    train_start = time.time()
    wf_processor.read_train_data()
    print("read training data took %.1f seconds" % ((time.time()-train_start)))
    train_start = time.time()
    wf_processor.train_model(wf_processor.data)
    print("training took %.1f seconds" % ((time.time()-train_start)))

    print('The whole workflow processing took %.1f seconds'%(time.time()-very_start))
