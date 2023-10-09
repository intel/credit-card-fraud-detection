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
import os
import argparse

from script import run_gnn_wf_docker

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
            "--config-path",
            required=False,
            type=str,
            default=None,
            help="specify the config file name")
    
    args, _ = parser.parse_known_args()
    if args.config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    else:
        config_path = args.config_path
    os.environ["CONFIG_PATH"] = config_path
    os.environ["DATA_IN_FILE"] = os.path.join(args.input_path, "processed_data.csv")
    os.environ["DATA_OUT"] = args.output_path
    os.environ["WORKSPACE"] = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.environ["DATA_OUT"]):
        os.mkdir(os.environ["DATA_OUT"])
    run_gnn_wf_docker.run(config_path=os.path.join(config_path, "gnn_analytics.yaml"))