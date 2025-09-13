from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import joblib
import time
import os

# Load the dataset
dataset = load_dataset("go_emotions", "simplified")   # It returnes a dictionary with 'train', 'validation', and 'test' splits

# Preprocess
df = pd.DataFrame(dataset['train'])  # Convert to DataFrame for easier manipulation(text, labels)
df = df.sample(1000, random_state=42)  # Use only 1000 samples
emotions = df['labels'].explode().unique()
mlb = MultiLabelBinarizer(classes=emotions, sparse_output=False)
labels = mlb.fit_transform(df['labels'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

inputs = preprocess_function(df['text'].tolist())
print(type(inputs['input_ids'])) 
print(df['text'].dtype)  

dataset = TensorDataset(
    inputs['input_ids'].type(torch.long), 
    inputs['attention_mask'].type(torch.long), 
    torch.tensor(labels, dtype=torch.float)
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# Function to get the next version number
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


# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotions), problem_type="multi_label_classification")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Training loop with GPU memory management
torch.cuda.empty_cache()
start_time = time.time()
for epoch in range(3):
    model.train()
    print(f"Starting epoch {epoch+1}")
    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        # print(input_ids.dtype, attention_mask.dtype, labels.dtype) 
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())

        # Compute the loss and its gradients
        loss = outputs.loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Zero your gradients for every batch
        optimizer.zero_grad()  

    # Save versioned model after each epoch
    version = get_next_version('./backend/data/saved_models')
    version_dir = os.path.join('./backend/data/saved_models', f'emotion_model_v{version}')
    os.makedirs(version_dir, exist_ok=True)
    model.save_pretrained(version_dir)
    tokenizer.save_pretrained(version_dir)
    joblib.dump(mlb, os.path.join(version_dir, 'mlb.pkl'))
    print(f"Saved model version {version} after epoch {epoch+1} to {version_dir}")


end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

# # Save the model
# import joblib
# model.save_pretrained('./backend/emotion_model')
# tokenizer.save_pretrained('../backend/emotion_model')
# joblib.dump(mlb, './backend/emotion_model/mlb.pkl')