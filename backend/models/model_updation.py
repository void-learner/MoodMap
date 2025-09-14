from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import joblib
import time
import os
import sqlite3

# Database connection and table creation
conn = sqlite3.connect('./backend/data/feedback.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedback
             (text TEXT, true_labels TEXT, predicted_labels TEXT)''')
conn.commit()

def add_feedback(text, true_labels=None, predicted_labels=None, is_correct=False):
    if not is_correct and true_labels:
        c.execute("INSERT INTO feedback (text, true_labels, predicted_labels) VALUES (?, ?, ?)",
                (text, ','.join(true_labels), ','.join(predicted_labels)))
        conn.commit()

    if c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] > 10:
        update_model()

# Function to get next version number
def get_next_version(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    existing_versions = [
        d for d in os.listdir(dir_path)
        if d.startswith('emotion_model_v') and os.path.isdir(os.path.join(dir_path, d))
    ]
    # handling case when no versions exist
    if not existing_versions:
        return 1
    max_version = max(
        int((v.split('_v')[-1])) for v in existing_versions
    )
    return max_version + 1
    
def get_existing_version(saved_dir_path):
    existing_versions = [
        d for d in os.listdir(saved_dir_path)
        if d.startswith('emotion_model_v') and os.path.isdir(os.path.join(saved_dir_path, d))
    ]

    if not existing_versions:
        raise ValueError("No existing model versions found.")
    max_version = max(int((v.split('_v')[-1])) for v in existing_versions)
    return os.path.join(saved_dir_path, f'emotion_model_v{max_version}')

def update_model():
    start_time = time.time()
    saved_dir_path = './backend/data/saved_models'

    latest_version = get_existing_version(saved_dir_path)

    model = BertForSequenceClassification.from_pretrained(os.path.join(latest_version, 'model'), problem_type="multi_label_classification")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(latest_version, 'tokenizer'))
    mlb = joblib.load(os.path.join(latest_version, 'mlb.pkl'))

    # Load feedback data from the database
    c.execute("SELECT * FROM feedback")
    feedback_data = c.fetchall()
    if not feedback_data:
        print("No feedback data available for model update.")
        return
    
    df = pd.DataFrame(feedback_data, columns=['text', 'true_labels', 'predicted_labels'])
    