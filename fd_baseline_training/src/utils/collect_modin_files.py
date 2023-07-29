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

import pandas as pd 
import subprocess
import glob 
import os
import sys 
import csv    
import yaml 
import time
import shlex

if __name__ == "__main__":
    start = time.time()
    wf_config_file = sys.argv[1]
    wf_config_file = os.path.abspath(wf_config_file)
    mode = sys.argv[2]

    if mode == "1":
        config_path = '/workspace/configs'
        with open(os.path.join(config_path, os.path.basename(wf_config_file)),'r') as file:
            config = yaml.safe_load(file)
        output_data_path = '/workspace/data/' + config['data_preprocess']['output_data_path']

    else:
        with open(wf_config_file,'r') as file:
            config = yaml.safe_load(file)
        output_data_path = os.path.join(config['env']['data_path'], config['data_preprocess']['output_data_path']) 

    worker_ips = config['env']['node_ips'][1:]
    file_name = 'processed_data.csv'

    for i, ip in enumerate(worker_ips):
        new_file_name = file_name.split('.')[0] +'_'+ str(i+1) + '.' + file_name.split('.')[1]
        src_path = f"{ip}:{output_data_path}/{file_name}"
        dest_path = f"{output_data_path}/{new_file_name}"
        command = f"scp -o StrictHostKeyChecking=no {shlex.quote(src_path)} {shlex.quote(dest_path)}"
        subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

    time.sleep(5)
    files = sorted(glob.glob(f'{output_data_path}/*.csv'))
    print(files)

    file_with_names = ''

    for file in files: 
        with open(file, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
        if 'year' in fieldnames:
            col_names = fieldnames
            file_with_names = file 
    
    print(col_names)
    
    df = []
    for file in files:
        if file == file_with_names:
            csv = pd.read_csv(file)
        else:
            csv = pd.read_csv(file, header=None, names=col_names)
        df.append(csv)

    data = pd.concat(df)
    print(data.shape)

    for f in files:
        os.remove(f)

    data.to_csv(f"{output_data_path}/{file_name}", index=False)
    print("this script took %.1f seconds" % ((time.time()-start)))

