import os
import shutil
import yaml
import json

with open('config.yml','r') as f:
    configs = yaml.safe_load(f)

if not os.path.exists(configs['data']):
    if not os.path.exists(configs['source_path']):
        print(f'数据集不存在or未准备好：{configs["source_path"]}')
    shutil.copytree(configs['source_path'], configs['data'])
if not os.path.exists(configs['model_path']):
    os.makedirs(configs['model_path'])
