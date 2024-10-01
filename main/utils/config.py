import os
import shutil
import sys
import yaml


def __docs__():

    '''
    default: ->

    device: "CPU"
    num_classes: 75
    image_size: 224
    batch_size: 16
    capacity: 250
    precision: "float16"
    learning_rate: 0.001
    epochs: 100
    static: "app/templates/static"
    source_path: "data"
    model_suffix: "-br.keras"
    model_path: "main/models"
    log_dir: "main/model/log"
    data: "TEMP/data"
    upload_dir: "TEMP/upload"
    train_csv: "TEMP/data/Training_set.csv"
    test_csv: "TEMP/data/Testing_set.csv"
    train_data: "TEMP/data/train"
    test_data: "TEMP/data/test"
    '''

def load_config(config_file='config.yml')->dict:
    with open(config_file,'r') as f:
        configs = yaml.safe_load(f)
    # ensure dirs
    if not os.path.exists(configs['data']):
        if not os.path.exists(configs['source_path']):
            print(f'数据集不存在or未准备好：{configs["source_path"]}')
        shutil.copytree(configs['source_path'],configs['data'])
    if not os.path.exists(configs['model_path']):
        os.system(f"mkdir {configs['model_path']}")
    return configs

if __name__=='__main__':
    load_config()
