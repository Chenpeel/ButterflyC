import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from utils.prepare import configs
import json
from flask import Flask, request, jsonify
import os

weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, configs['num_classes'])

model_save_path = os.path.join(configs['models'], configs['model_name'])
model.load_state_dict(torch.load(model_save_path))
model.eval()

with open(os.path.join(configs['log'],'label_mapping.json'), 'r') as f:
    label_mapping = json.load(f)
index_to_label = {v: k for k, v in label_mapping.items()}


transform = transforms.Compose([
    transforms.Resize((configs['image_size'], configs['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = index_to_label[predicted.item()]
    return predicted_label
