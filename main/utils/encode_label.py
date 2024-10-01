from sklearn.preprocessing import LabelEncoder
import pandas as pd
from main.utils.config import load_config

configs = load_config()

label_encoder = LabelEncoder()

def fit_label_encoder():
    train_csv = pd.read_csv(configs['train_csv'])
    train_csv.columns = train_csv.columns.str.strip()
    label_encoder.fit(train_csv['label'])

def encode_label(label):
    return label_encoder.transform([label])[0]
fit_label_encoder()
