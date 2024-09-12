import os
import json
import yaml
import tqdm

def load_configs():
    with open("configs.yaml", "r") as f:
        configs = yaml.safe_load(f)

    configs['train_path'] = configs['train_path'].replace('{{dataset}}', configs['dataset'])
    configs['valid_path'] = configs['valid_path'].replace('{{dataset}}', configs['dataset'])
    configs['classes_path'] = configs['classes_path'].replace('{{dataset}}', configs['dataset'])
    configs['classes_path'] = configs['classes_path'].replace('{{classes_filename}}', configs['classes_filename'])

    return configs

def json2dict(json_path):
    with open(json_path, "r") as f:
        cat2name = json.load(f)
    return cat2name

def label_files(work_dir, json_path,data_dir):
    images = []
    labels = []
    cat2name=json2dict(json_path)
    n = len(cat2name)+1
    create_work_dir(work_dir,data_dir)
    print("labeling files...")
    for name in tqdm.tqdm(range(1,n)):
        for image in os.listdir(os.path.join(work_dir, str(name))):
            if image.endswith(".jpg"):
                images.append(os.path.join(work_dir, str(name), image))
                labels.append(str(name-1))
    return images, labels

def create_work_dir(work_dir,data_dir):
    if os.path.exists(work_dir):
        os.system(f'rm -rf {work_dir}')
    os.makedirs(work_dir)
    os.system(f'cp -r {data_dir}/* {work_dir}/')
    if os.name == 'posix':
        os.system(f'rm -rf {work_dir}/.DS_Store')
    return work_dir