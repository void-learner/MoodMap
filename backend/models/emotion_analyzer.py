from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import joblib
import time
import os


def train_emotion_model():
    # Load the dataset
    dataset = load_dataset("go_emotions", "simplified")   # It returnes a dictionary with 'train', 'validation', and 'test' splits
    emotion_names = dataset['train'].features['labels'].feature.names

    # Preprocess
    df = pd.DataFrame(dataset['train'])  # Convert to DataFrame for easier manipulation(text, labels)
    df = df.sample(18000, random_state=42)  # Use only 18000 samples
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

    # Compute class weights based on label frequency
    label_counts = labels.sum(axis=0)
    class_weights = 1.0 / (label_counts + 1e-5)

    # Nomalize
    class_weights = class_weights / class_weights.max()

    criterion = BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    for epoch in range(1):
        model.train()
        print(f"Starting epoch {epoch+1}")
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            # print(input_ids.dtype, attention_mask.dtype, labels.dtype) 
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())

            # logistics
            logits = outputs.logits

            # Compute the loss and its gradients
            loss = criterion(logits, labels.float())
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        joblib.dump(emotion_names, os.path.join(version_dir, 'emotion_names.pkl'))
        print(f"Saved model version {version} after epoch {epoch+1} to {version_dir}")


    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")


def analyze_emotion(text):
    saved_dir_path = './backend/data/saved_models'
    
    if not os.path.exists(saved_dir_path) or not os.listdir(saved_dir_path):
        raise ValueError("No saved models found. Please train the model first.")
    
    existing_versions = [
        d for d in os.listdir(saved_dir_path)
        if d.startswith('emotion_model_v') and os.path.isdir(os.path.join(saved_dir_path, d))
    ]

    if not existing_versions:
        raise ValueError("No valid model versions found in the saved models directory.")
    
    latest_version = max(
        int((v.split('_v')[-1])) for v in existing_versions
    )
    version_dir = os.path.join(saved_dir_path, f'emotion_model_v{latest_version}')
    model = BertForSequenceClassification.from_pretrained(version_dir)
    tokenizer = BertTokenizer.from_pretrained(version_dir)
    mlb = joblib.load(os.path.join(version_dir, 'mlb.pkl'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()   # putting model into evaluation mode

    # load emotion names
    emotion_names = joblib.load(os.path.join(version_dir, 'emotion_names.pkl'))

    '''During inference, it applies those patterns to make decisions.'''
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    for key in inputs:
        inputs[key] = inputs[key].to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits  # classification model outputs
    probabilities = torch.sigmoid(outputs)
    threshold = 0.2
    predicted_prob = (probabilities >= threshold).cpu().numpy()[0]   # gives binary vector
    emotions = mlb.classes_

    # Return all with probability above threshold
    emotions_with_probs = [
        {'label': emotion_names[idx], 'probability': float(prob)} for idx, prob in enumerate(predicted_prob) if prob >= threshold
    ]

    emotions_with_probs.sort(key=lambda x: x['probability'], reverse=True)

    return emotions_with_probs



# Test samples
# test_samples = [
#     {"text": "LETS FUCKING GOOOOO", "expected": ["excitement"]},  
#     {"text": "*aggressively tells friend I love them*", "expected": ["love"]},  
#     {"text": "you almost blew my fucking mind there.", "expected": ["surprise", "admiration"]},  
#     {"text": "daaaaaamn girl!", "expected": ["admiration"]},  
#     {"text": "[NAME] wept.", "expected": ["surprise"]},  
#     {"text": "I try my damndest. Hard to be sad these days when I got this guy with me", "expected": ["joy"]},  
#     {"text": "hell yeah my brother", "expected": ["approval"]},  
#     {"text": "[NAME] is bae, how dare you.", "expected": ["love"]},  
#     {"text": "I'm so happy today!", "expected": ["joy"]},  
#     {"text": "I feel really sad.", "expected": ["sadness"]},  
#     {"text": "This is neutral.", "expected": ["neutral"]}  
# ]

# print("\nManual Test Results:")
# for sample in test_samples:
#     predicted = analyze_emotion(sample['text'])
#     print(f"Text: '{sample['text']}'")
#     print(f"Predicted: {[p['label'] for p in predicted]}")
#     print(f"Expected (approx.): {sample['expected']}")
#     print("---")

