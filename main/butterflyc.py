import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from utils.prepare import configs
import json
import numpy as np

class normal_data(Dataset):
    def __init__(self, csv, data_dir, transform=None) -> None:
        self.data_frame = pd.read_csv(csv)
        self.root_dir = data_dir
        self.transform = transform
        self.labels, self.label_to_index = pd.factorize(self.data_frame.iloc[:, 1])
        self.num_classes = len(self.label_to_index)

        # 保存标签映射为JSON文件
        with open(os.path.join(configs['log'],'label_mapping.json'), 'w') as f:
            json.dump({label: int(index) for index, label in enumerate(self.label_to_index)}, f)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        img_name = self.data_frame.iloc[index, 0]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        # 转换为one-hot编码
        label_one_hot = np.zeros(self.num_classes)
        label_one_hot[label] = 1
        return image, torch.tensor(label_one_hot, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((configs['image_size'], configs['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = normal_data(csv=configs['train_csv'],
                            data_dir=configs['train_data'],
                            transform=transform)
train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, configs['num_classes'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'])

model.to(configs['device'])

for epoch in range(configs['epochs']):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{configs['epochs']}")

            inputs = inputs.to(configs['device'])
            labels = labels.to(configs['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

            tepoch.set_postfix(loss=running_loss / total, accuracy=100. * correct / total)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch + 1}/{configs['epochs']}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


torch.save(model.state_dict(), os.path.join(configs['models'],configs['model_name']))
