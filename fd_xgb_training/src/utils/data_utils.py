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

import glob 
import os 
import sys 
from pathlib import Path 
import shutil
import numpy as np 


def read_csv_files(raw_data_path, engine, ignore_cols=None):
    if engine=='pandas':
        import pandas as pd 
    elif engine=='modin':
        import modin.pandas as pd 
    else:
        raise ValueError('Engine can either be pandas or modin.')
    
    files = glob.glob(f'{raw_data_path}/*.csv')
    df = []
    for file in files: 
        csv = pd.read_csv(file)
        if ignore_cols is not None:
            print("dropping columns...")
            csv.drop(columns=ignore_cols, inplace=True)
        else:
            print("reading without dropping columns...")
        df.append(csv)
    data = pd.concat(df) 
    print(f"data has the shape {data.shape}")
    return data 


def make_dir(path):
    path = Path(path)

    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    
    os.makedirs(path)
    
    return None


def divide_save_df(df, save_format, save_data_path, num_partitions):
    make_dir(save_data_path)
    
    if save_format == 'csv':
        df_splits = np.array_split(df, num_partitions)
        for i, data in enumerate(df_splits):
            data.to_csv(f"{save_data_path}/partition_{i}.csv", index=False)
    else:
        print("other data format not supported")
    


def has_dir(data_path, folder_name):

    for fname in os.listdir(data_path):
        if fname==folder_name and os.path.isdir(os.path.join(data_path,fname)):
            return True 
    
    return False 


def read_parquet_spark(spark, data_path):

    data = spark.read.parquet(data_path)
    print(f'({data.count()}, {len(data.columns)})')
    
    return data
