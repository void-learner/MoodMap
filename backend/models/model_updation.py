from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
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
        os.makedirs(dir_path)
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

    # Load feedback dataset
    c.execute("SELECT COUNT(*) FROM feedback")
    rows = c.fetchall()  # rows is now a list of tuples
    count = c.fetchone()[0]
    if not rows or count < 10:
        print("Not enough feedback data to update the model.")
        return

    latest_version = get_existing_version(saved_dir_path)
    version_dir = latest_version
    model = BertForSequenceClassification.from_pretrained(version_dir, problem_type="multi_label_classification")
    tokenizer = BertTokenizer.from_pretrained(version_dir)
    mlb = joblib.load(os.path.join(version_dir, 'mlb.pkl'))

    # Load feedback data from the database
    c.execute("SELECT * FROM feedback")
    feedback_data = c.fetchall()
    if not feedback_data:
        print("No feedback data available for model update.")
        return
    
    df = pd.DataFrame(feedback_data, columns=['text', 'true_labels', 'predicted_labels'])
    
    # Preprocess the data
    new_labels = df['true_labels'].apply(lambda x: x.split(',') if x else [])
    new_inputs = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    new_dataset = TensorDataset(
        new_inputs['input_ids'].type(torch.long),
        new_inputs['attention_mask'].type(torch.long),
        torch.tensor(mlb.transform(new_labels), dtype=torch.float)
    )
    new_loader = DataLoader(new_dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.to(device)

    # Training loop
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for epoch in range(3):
        print(f"Starting of epoch{epoch+1}")
        for batches in new_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batches]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())

            # Forward pass
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save the updated model
    new_version = get_next_version(saved_dir_path)
    new_version_dir = os.path.join(saved_dir_path, f"emotion_model_v{new_version}")
    os.makedirs(new_version_dir, exist_ok=True)
    model.save_pretrained(new_version_dir)
    tokenizer.save_pretrained(new_version_dir)
    joblib.dump(mlb, os.path.join(new_version_dir, 'mlb.pkl'))
    print(f"Model updated and saved to {new_version_dir}")

    # Clear feedback table
    c.execute("DELETE FROM feedback")
    conn.commit()
    conn.close()
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
