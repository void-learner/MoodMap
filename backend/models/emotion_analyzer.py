from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load the dataset
dataset = load_dataset("go_emotions", "simplified")   # It returnes a dictionary with 'train', 'validation', and 'test' splits

# Preprocess
df = pd.DataFrame(dataset['train'])  # Convert to DataFrame for easier manipulation(text, labels)
emotions = df['labels'].explode().unique()
mlb = MultiLabelBinarizer(classes=emotions, sparse_output=False)
labels = mlb.fit_transform(df['labels'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(text):
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

inputs = preprocess_function(df['text'].tolist())
# print(type(inputs['input_ids'])) 
# print(df['text'].dtype)  

dataset = TensorDataset(
    inputs['input_ids'].type(torch.long), 
    inputs['attention_mask'].type(torch.long), 
    torch.tensor(labels, dtype=torch.float)
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(emotions), problem_type="multi_label_classification")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop with GPU memory management
torch.cuda.empty_cache()
for epoch in range(3):
    model.train()
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

# Save the model
import joblib
model.save_pretrained('./emotion_model')
tokenizer.save_pretrained('./emotion_model')
joblib.dump(mlb, './emotion_model/mlb.pkl')